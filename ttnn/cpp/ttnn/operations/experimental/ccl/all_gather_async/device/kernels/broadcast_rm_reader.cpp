// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t row_size = get_compile_time_arg_val(2);
constexpr uint32_t num_rows_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t num_packets_per_page = get_compile_time_arg_val(4);
constexpr uint32_t max_packet_size = get_compile_time_arg_val(5);
constexpr uint32_t is_sender = get_compile_time_arg_val(6);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    DPRINT << "reader page_size" << page_size << ENDL();
    // DPRINT << "reader row_size" << row_size << ENDL();
    // DPRINT << "reader num_packets_per_page" << num_packets_per_page << ENDL();
    if (is_sender) {
        size_t arg_idx = 0;
        // Load the input tensor spec
        address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
        uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
        uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);

        DPRINT << "reader row_id_start" << row_id_start << ENDL();
        DPRINT << "reader row_id_end" << row_id_end << ENDL();

        // typedef ShardedInfo<
        //     get_compile_time_arg_val(7),
        //     get_compile_time_arg_val(8),
        //     get_compile_time_arg_val(9),
        //     get_compile_time_arg_val(10),
        //     get_compile_time_arg_val(11),
        //     get_compile_time_arg_val(12),
        //     get_compile_time_arg_val(13)>
        //     tensor_shard_info;

        // const auto [mapping_table, rt_increment] =
        //     experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx++));
        // experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        //     .bank_base_address = tensor_address0, .shard_array = mapping_table};

        constexpr auto tensor0_args = TensorAccessorArgs<7>();
        auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0, row_size);

        // uint32_t row_id = row_id_start;
        for (uint32_t row_id = row_id_start; row_id < row_id_end; row_id++) {
            DPRINT << "reader row_id" << row_id << ENDL();
            // DPRINT << "reader num_rows_per_packet" << num_rows_per_packet << ENDL();
            cb_reserve_back(cb0_id, 1);
            uint32_t l1_write_addr = get_write_ptr(cb0_id);

            uint64_t noc_src_addr = tensor0_addrgen.get_noc_addr(row_id, 0);
            noc_async_read(noc_src_addr, l1_write_addr, page_size);
            noc_async_read_barrier();

            // for (uint32_t i = 0; i < num_rows_per_packet && row_id < row_id_end; ++i) {
            //     uint64_t noc_src_addr = tensor0_addrgen.get_noc_addr(row_id);
            //     uint32_t bytes_remaining = page_size;
            //     uint32_t offset = 0;
            //     for (uint32_t pkt = 0; pkt < num_packets_per_page && bytes_remaining > 0; ++pkt) {
            //         uint64_t noc_src_addr = tensor0_addrgen.get_noc_addr(row_id, offset);
            //         uint32_t packet_size = std::min(max_packet_size, bytes_remaining);
            //         noc_async_read(noc_src_addr, l1_write_addr + offset, packet_size);
            //         // DPRINT << "reader not_read here pkt>" << pkt << ENDL();
            //         offset += packet_size;
            //         bytes_remaining -= packet_size;
            //     }
            //     l1_write_addr += page_size;
            //     row_id++;
            // }
            // noc_async_read_barrier();

            cb_push_back(cb0_id, 1);
        }
    }
    DPRINT << "exiting reader kernel" << ENDL();
}
