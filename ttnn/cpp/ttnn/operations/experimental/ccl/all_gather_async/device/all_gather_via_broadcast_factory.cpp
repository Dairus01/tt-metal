// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_via_broadcast_factory.hpp"

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation_types.hpp"

namespace ttnn {

using namespace ccl;

namespace experimental::prim {

AllGatherViaBroadcastFactory::cached_mesh_workload_t AllGatherViaBroadcastFactory::create_mesh_workload(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
    printf("hello here-> all_gather_via_broadcast\n");
    // tt::tt_metal::distributed::MeshWorkload workload;
    // std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    // for (const auto& coord : tensor_coords.coords()) {
    //     auto cached_program = create_at(operation_attributes, coord, tensor_args, output_tensor);
    //     workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
    //     shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    // }
    // return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args.input_tensor,
            output_tensor,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherViaBroadcastFactory::cached_program_t AllGatherViaBroadcastFactory::create_at(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const Tensor& input,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = input;
    tt::tt_metal::Program program{};

    auto* mesh_device = input_tensor.device();

    uint32_t ring_size = operation_attributes.ring_size;
    uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    printf("%u/%u\n", ring_index, ring_size);
    std::cout << "intput: " << input_tensor.logical_shape() << std::endl;
    std::cout << "output: " << output_tensor.logical_shape() << std::endl;
    std::cout << "dim: " << operation_attributes.dim << std::endl;
    std::cout << "(input) page_size: " << input_tensor.buffer()->aligned_page_size() << std::endl;
    std::cout << "(input) page_size (non-aligned): " << input_tensor.buffer()->page_size() << std::endl;
    std::cout << "(output) page_size: " << output_tensor.buffer()->aligned_page_size() << std::endl;
    std::cout << "(output) page_size (non-aligned): " << output_tensor.buffer()->page_size() << std::endl;
    std::cout << "memory config: " << output_tensor.memory_config() << '\n';

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    bool tilized = output_tensor.layout() == ttnn::TILE_LAYOUT;

    // std::cout << "memory config: " << input_tensor.memory_config() << '\n';
    sharded = false;
    tilized = false;

    uint32_t num_width_shards = 1;
    if (!tilized && (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                     output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED)) {
        num_width_shards = output_tensor.padded_shape()[-1] / output_tensor.memory_config().shard_spec()->shape[1];
        // std::cout << "here shard_spec " << output_tensor.memory_config() << '\n';
    }
    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
        ring_size, ring_index, operation_attributes.topology, true);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] = ::ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_workers_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        CoreCoord(0, 0),
        std::nullopt);

    // Info for RM tensors
    // uint32_t row_size = input_tensor.logical_shape()[-1] * input_tensor.element_size();
    const uint32_t page_size = input_tensor.buffer()->aligned_page_size();

    uint32_t num_rows = input_tensor.logical_shape().size() > 2
                            ? input_tensor.logical_shape()[-2] * input_tensor.logical_shape()[-3]
                            : input_tensor.logical_shape()[-2];
    if (input_tensor.logical_shape().size() == 4) {
        num_rows *= input_tensor.logical_shape()[0];
    }

    // L1 Scratch CB Creation
    DataType dtype = input_tensor.dtype();
    const uint32_t fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t MAX_PACKET_SIZE_BYTES =
        dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;
    const size_t packet_size_bytes =
        tilized ? tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes() : MAX_PACKET_SIZE_BYTES;
    std::cout << "packet_size_bytes " << packet_size_bytes << '\n';
    size_t max_packet_size = packet_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    std::cout << "num_pages_per_packet " << num_pages_per_packet << '\n';
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);

    uint32_t buffer_page_size = page_size;
    uint32_t num_packets_per_page =
        static_cast<uint32_t>(std::ceil(static_cast<double>(buffer_page_size) / max_packet_size));

    // if (!tilized) {
    //     if (num_width_shards > 1) {
    //         buffer_page_size = input_tensor.memory_config().shard_spec()->shape[1] * input_tensor.element_size();
    //     }

    //     // uint32_t num_rows_per_packet = (max_packet_size / buffer_page_size >= 2) ? 2 : 1;
    //     cb_src0_config =
    //         tt::tt_metal::CircularBufferConfig(3 * buffer_page_size, {{src0_cb_index, df}})
    //             .set_page_size(src0_cb_index, buffer_page_size);
    // }

    // cb_src0_config = tt::tt_metal::CircularBufferConfig(3 * page_size, {{src0_cb_index, df}})
    //                      .set_page_size(src0_cb_index, page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,                                      // cb0_id
        buffer_page_size,                                   // page_size
        input_tensor.buffer()->aligned_page_size(),         // row_size
        (max_packet_size / buffer_page_size >= 2) ? 2 : 1,  // num_rows_per_packet
        num_packets_per_page,                               // num_packets_per_page
        max_packet_size,
        true,  // is_sender
    };

    // Writer kernel
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,                                // cb0_id
        num_pages_per_packet,                         // packet_size_in_pages
        output_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
        num_targets_forward,                          // num_targets_forward_direction
        num_targets_backward,                         // num_targets_backward_direction
        true,                                         // is_sender
    };

    if (!tilized) {
        writer_compile_args = {
            src0_cb_index,  // cb0_id
            buffer_page_size,
            output_tensor.buffer()->aligned_page_size(),  // row_size
            max_packet_size,
            1,
            // (max_packet_size / buffer_page_size >= 2) ? 2 : 1,  // num_rows_per_packet
            num_packets_per_page,  // num_packets_per_row
            num_targets_forward,   // num_targets_forward_direction
            num_targets_backward,  // num_targets_backward_direction
            true,                  // is_sender
        };
    }
    std::vector<uint32_t> mcast_forward_args(2, 0);
    std::vector<uint32_t> mcast_backward_args(2, 0);
    if (forward_coord.has_value()) {
        mcast_forward_args[0] = 1;
        mcast_forward_args[1] = num_targets_forward;
    }
    if (backward_coord.has_value()) {
        mcast_backward_args[0] = 1;
        mcast_backward_args[1] = num_targets_backward;
    }
    writer_compile_args.insert(writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input_tensor, reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(input_tensor, writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_reader.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, kernel_defines));

    // Writer
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, kernel_defines));

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / operation_attributes.num_links;
        if (!tilized) {
            base_pages_per_worker = num_rows / operation_attributes.num_links;
        }
        uint32_t remainder = input_tensor_num_pages % operation_attributes.num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
            input_tile_id_start,               // tile_id_start
            input_tile_id_end,                 // tile_id_end
        };

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = ring_size * operation_attributes.num_links;
        uint32_t output_tile_id_start = input_tile_id_start;
        uint32_t output_tile_id_end = input_tile_id_end;
        uint32_t num_result_rows = 1u;
        for (int i = operation_attributes.dim; i < input_tensor.padded_shape().size() - 1; ++i) {
            num_result_rows *= input_tensor.padded_shape()[i];
        }
        uint32_t write_offset = num_width_shards * num_result_rows * ring_index;
        printf(
            "num_rows %u; ring_index %u; write_offset %u, num_width_shards %u\n",
            num_result_rows,
            ring_index,
            write_offset,
            num_width_shards);
        // aligned_page into tensoraccessor
        (void)output_tile_id_start;
        (void)output_tile_id_end;

        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // tensor_address0  //HERE
            write_offset,                       //
            semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
            num_rows,
            // output_tile_id_start * num_width_shards,  // tile_id_start
            // output_tile_id_end * num_width_shards,    // tile_id_end
            wait_output_semaphore,        // wait_output_semaphore
            reset_global_semaphore,       // reset_global_semaphore
            drain_sync_core.x,            // out_ready_sem_noc0_x
            drain_sync_core.y,            // out_ready_sem_noc0_y
            out_ready_sem_wait_value,     // out_ready_sem_wait_value
            barrier_semaphore.address(),  // barrier_sem
            barrier_core.x,               // barrier_sem_noc0_x
            barrier_core.y                // barrier_sem_noc0_y
        };
        auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
        writer_rt_args.push_back(num_connections);

        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
        dst_nodes.reserve(num_connections);
        if (forward_coord.has_value()) {
            const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            dst_nodes.push_back(forward_coord_fabric_node_id);
        }
        if (backward_coord.has_value()) {
            const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            dst_nodes.push_back(backward_coord_fabric_node_id);
        }

        append_routing_plane_connection_manager_rt_args(
            sender_fabric_node_id, dst_nodes, {link}, program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = sender_worker_cores,
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .semaphore = semaphore,
        .barrier_semaphore = barrier_semaphore,
        .ring_index = ring_index,
    };

    return {std::move(program), std::move(shared_variables)};
}

void all_gather_async_minimal_default_helper_override_runtime_arguments_2(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::vector<GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& output);

void all_gather_async_minimal_default_helper_override_runtime_arguments_2(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::vector<GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& output) {
    // Update runtime arguments for all worker cores
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                uint32_t mux_core_offset = (link * num_cores_per_link) +
                                           (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                tt::tt_metal::CoreCoord core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

                const auto& out_ready_semaphore = semaphore.at(dir);

                // sender reader
                auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = output.buffer()->address();
                worker_reader_sender_runtime_args[2] = out_ready_semaphore.address();

                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = output.buffer()->address();
                worker_writer_sender_runtime_args[3] = out_ready_semaphore.address();

                if (barrier_semaphore.has_value()) {
                    worker_writer_sender_runtime_args[5] = barrier_semaphore.value().address();
                }
            }
        }
    }
}

void AllGatherViaBroadcastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
    (void)cached_workload;
    (void)operation_attributes;
    (void)tensor_args;
    (void)output_tensor;

    // // Update runtime arguments for each program in the mesh workload
    // for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
    //     auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

    //     const auto& input = tensor_args.input_tensor;
    //     const auto& output = output_tensor;

    //     auto semaphore = operation_attributes.semaphore;
    //     auto barrier_semaphore = operation_attributes.barrier_semaphore;

    //     all_gather_async_minimal_default_helper_override_runtime_arguments_2(
    //         program,
    //         shared_vars.reader_kernel_id,
    //         shared_vars.writer_kernel_id,
    //         shared_vars.all_cores,
    //         operation_attributes.num_links,
    //         shared_vars.num_directions_per_link,
    //         shared_vars.num_workers_per_direction,
    //         shared_vars.num_mux_cores_per_direction_per_link,
    //         shared_vars.num_cores_per_link,
    //         barrier_semaphore,
    //         semaphore,
    //         input,
    //         output);
    // }
}

}  // namespace experimental::prim
}  // namespace ttnn
