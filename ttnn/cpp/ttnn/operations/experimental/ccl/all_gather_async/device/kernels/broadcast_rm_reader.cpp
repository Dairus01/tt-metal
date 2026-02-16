// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

using address_t = uint32_t;

void kernel_main() {
    // DPRINT << "reader page_size" << page_size << ENDL();
    // DPRINT << "reader row_size" << row_size << ENDL();
    // DPRINT << "reader num_packets_per_page" << num_packets_per_page << ENDL();

    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(2);
    constexpr auto tensor0_args = TensorAccessorArgs<3>();
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);

    // DPRINT << "reader row_id_start" << row_id_start << ENDL();
    // DPRINT << "reader row_id_end" << row_id_end << ENDL();

    TensorAccessor tensor0_addrgen(tensor0_args, tensor_address0, page_size);

    uint32_t bytes_in_cb_page = 0;
    for (uint32_t row_id = row_id_start; row_id < row_id_end; row_id++) {
        // DPRINT << "reader row_id " << row_id << ENDL();

        if (bytes_in_cb_page == 0) {
            cb_reserve_back(cb0_id, 1);
        }

        uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t noc_src_addr = tensor0_addrgen.get_noc_addr(row_id, 0);
        noc_async_read(noc_src_addr, l1_write_addr + bytes_in_cb_page, page_size);
        bytes_in_cb_page += page_size;

        if (bytes_in_cb_page == cb_page_size) {
            noc_async_read_barrier();
            cb_push_back(cb0_id, 1);
            bytes_in_cb_page = 0;
        }
    }
}
