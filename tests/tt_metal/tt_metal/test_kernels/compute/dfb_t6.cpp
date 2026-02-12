// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t num_entries = get_compile_time_arg_val(0);

    experimental::DataflowBuffer dfb_in(0);
    experimental::DataflowBuffer dfb_out(1);

    // DPRINT << "producer_idx: " << producer_idx << " num_entries_per_producer: " << num_entries_per_producer <<
    // ENDL();

    // for (uint32_t tensix_id = 0; tensix_id < 4; tensix_id++) {
    //     for (uint32_t tc_id = 0; tc_id < 16; tc_id++) {
    //         DPRINT << "tensix_id: " << tensix_id << " tc_id: " << tc_id
    //                << " capacity: " << static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id)) << ENDL();
    //     }
    // }

    for (uint32_t tile_id = 0; tile_id < num_entries; tile_id++) {
        // DPRINT << "rbw" << ENDL();
        dfb_in.wait_front(1);
        // do some processing on unpacker
        dfb_in.pop_front(1);

        dfb_out.reserve_back(1);
        // do some processing on packer
        dfb_out.push_back(1);
        // DPRINT << "pbd" << ENDL();
    }
    DPRINT << "PFW" << ENDL();
    dfb_out.finish();
    DPRINT << "PFD" << ENDL();
}
