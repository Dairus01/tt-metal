// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include "tt_metal/fabric/hw/inc/linear/api.h"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_page_size = get_compile_time_arg_val(1);
constexpr uint32_t out_page_size = get_compile_time_arg_val(2);
constexpr uint32_t packet_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr uint32_t is_sender = get_compile_time_arg_val(6);
constexpr uint32_t start_distance_in_hops_forward = get_compile_time_arg_val(7);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(8);
constexpr uint32_t start_distance_in_hops_backward = get_compile_time_arg_val(9);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(10);

inline constexpr uint32_t sharded_args_start_idx = 11;

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t write_page_offset = get_arg_val<uint32_t>(arg_idx++);
    const address_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    // uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
    // uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_output_pages = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto unicast_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto scatter_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    constexpr auto tensor0_args = TensorAccessorArgs<sharded_args_start_idx>();
    auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0, out_page_size);
    open_connections(fabric_connection, num_connections, arg_for_fab);
    uint8_t starts[] = {
        static_cast<uint8_t>(start_distance_in_hops_forward), static_cast<uint8_t>(start_distance_in_hops_backward)};
    uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }
    uint32_t payload_size = packet_size;
    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, unicast_route_id, starts, ranges, nullptr, payload_size);
    fabric_multicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        fabric_connection,
        scatter_route_id,
        starts,
        ranges,
        NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(payload_size)}),
        payload_size * 2);

    uint32_t num_total_targets = num_targets_forward_direction + num_targets_backward_direction;

    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts,
        ranges,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    // 1. mcast via fabric to remote tensor addresses

    if (is_sender) {
        uint32_t bytes_read_from_cb_page = cb_page_size;
        for (uint32_t page_id = 0; page_id < num_output_pages; ++page_id) {
            uint32_t out_page_id = page_id + write_page_offset;
            if (bytes_read_from_cb_page == cb_page_size) {
                cb_wait_front(cb0_id, 1);
                bytes_read_from_cb_page = 0;
            }
            auto l1_read_addr = get_read_ptr(cb0_id);
            auto out_page_addr = tensor0_addrgen.get_noc_addr(out_page_id, 0);

            for (uint32_t packet_offset = 0; packet_offset < out_page_size; packet_offset += packet_size) {
                // DPRINT << "packet_loop offset: " << packet_offset << ENDL();
                // first send
                auto packet_read_addr = l1_read_addr + bytes_read_from_cb_page + packet_offset;
                fabric_multicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    fabric_connection,
                    unicast_route_id,
                    packet_read_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{
                        linear::addrgen_detail::get_noc_address(tensor0_addrgen, out_page_id, packet_offset)},
                    packet_size);
            }
            noc_async_write(l1_read_addr + bytes_read_from_cb_page, out_page_addr, out_page_size);
            bytes_read_from_cb_page += out_page_size;

            if (bytes_read_from_cb_page == cb_page_size) {
                noc_async_writes_flushed();
                cb_pop_front(cb0_id, 1);
            }
        }

        // 2. mcast output ready semaphore
        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
        // increment locally
        uint64_t out_ready_sem_noc_addr =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
        noc_semaphore_inc(out_ready_sem_noc_addr, 1);

        // 3. wait for mcast output ready semaphore
        if (wait_output_semaphore) {
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
            noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
        }

        // 4. global semaphore reset
        if (reset_global_semaphore) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
        }

        close_connections(fabric_connection);

        noc_async_write_barrier();
    } else {
        if (wait_output_semaphore) {
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
            noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
        }

        if (reset_global_semaphore) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
        }

        close_connections(fabric_connection);

        noc_async_write_barrier();
    }
}
