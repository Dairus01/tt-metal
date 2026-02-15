// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host+device compatible header: pure data layout for datapath usage capture.
// No methods, no device-specific includes.
// All functions that operate on this struct are in fabric_trimming.hpp (device-only).

#include <array>
#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

// Primary template - enabled implementation (full data storage)
template <bool ENABLED, size_t NUM_VC = 2, size_t MAX_NUM_SENDER_CHANNELS = 9>
struct FabricDatapathUsageL1Results {
    using SenderChannelUsedBitfield = uint16_t;
    using ReceiverChannelDataForwardedBitfield = uint16_t;
    using NocSendTypeBitfield = uint16_t;

    // Record the max and min packet size seen by each sender channel
    std::array<uint16_t, MAX_NUM_SENDER_CHANNELS> sender_channel_min_packet_size_seen_bytes_by_vc = {};
    std::array<uint16_t, MAX_NUM_SENDER_CHANNELS> sender_channel_max_packet_size_seen_bytes_by_vc = {};

    // A bit is set high if the sender channel with ID matching that bit (offset) processed any traffic
    SenderChannelUsedBitfield sender_channel_used_bitfield_by_vc = {};

    // A bit is set high if the sender channel on this VC is forwarded to the receiver channel with ID matching
    // that bit (offset). Used only for validation after readback.
    std::array<SenderChannelUsedBitfield, NUM_VC> sender_channel_forwarded_to_bitfield_by_vc = {};

    // A bit is set high if the receiver channel with ID matching that bit (offset) has forwarded any traffic
    ReceiverChannelDataForwardedBitfield receiver_channel_data_forwarded_bitfield_by_vc = {};

    // A bit is set high if the receiver channel on this VC forwards a noc message of that type
    std::array<NocSendTypeBitfield, NUM_VC> used_noc_send_type_by_vc_bitfield = {};
};

// Specialization for disabled implementation - zero overhead, no storage
template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
struct FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS> {};

}  // namespace tt::tt_fabric
