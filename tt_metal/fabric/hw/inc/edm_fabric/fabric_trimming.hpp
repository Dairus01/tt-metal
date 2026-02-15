// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-only header: all functions that operate on FabricDatapathUsageL1Results.
// The core data structure is defined in fabric_trimming_types.hpp (host+device).
// Functions here take the data struct as an argument.

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming_types.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"  // for NocSendType

#include "internal/risc_attribs.h"

#include <limits>

namespace tt::tt_fabric {

// ============================================================================
// Free functions operating on FabricDatapathUsageL1Results<true> (enabled)
// ============================================================================

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_reset(FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r) {
    r.sender_channel_min_packet_size_seen_bytes_by_vc.fill(std::numeric_limits<uint16_t>::max());
    r.sender_channel_max_packet_size_seen_bytes_by_vc.fill(0);
    r.sender_channel_used_bitfield_by_vc = 0;
    r.sender_channel_forwarded_to_bitfield_by_vc.fill(0);
    r.receiver_channel_data_forwarded_bitfield_by_vc = 0;
    r.used_noc_send_type_by_vc_bitfield.fill(0);
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_sender_channel_used(
    FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r, size_t sender_channel_id) {
    r.sender_channel_used_bitfield_by_vc |= (1 << sender_channel_id);
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_receiver_channel_data_forwarded(
    FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r, size_t receiver_channel_id) {
    r.receiver_channel_data_forwarded_bitfield_by_vc |= (1 << receiver_channel_id);
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_update_sender_channel_packet_size(
    FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r,
    size_t sender_channel_id,
    uint16_t packet_size_bytes) {
    if (packet_size_bytes < r.sender_channel_min_packet_size_seen_bytes_by_vc[sender_channel_id]) {
        r.sender_channel_min_packet_size_seen_bytes_by_vc[sender_channel_id] = packet_size_bytes;
    }
    if (packet_size_bytes > r.sender_channel_max_packet_size_seen_bytes_by_vc[sender_channel_id]) {
        r.sender_channel_max_packet_size_seen_bytes_by_vc[sender_channel_id] = packet_size_bytes;
    }
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_sender_channel_forwarded_to(
    FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r,
    size_t vc_id,
    size_t receiver_channel_id) {
    r.sender_channel_forwarded_to_bitfield_by_vc[vc_id] |= (1 << receiver_channel_id);
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_merge_sender_channel_forwarded_to(
    FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r, size_t vc_id, uint16_t mask) {
    r.sender_channel_forwarded_to_bitfield_by_vc[vc_id] |= mask;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_noc_send_type_used(
    FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r,
    size_t vc_id,
    NocSendType noc_send_type) {
    r.used_noc_send_type_by_vc_bitfield[vc_id] |= (1 << static_cast<uint8_t>(noc_send_type));
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_sender_channel_used(
    const FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r, size_t sender_channel_id) {
    return (r.sender_channel_used_bitfield_by_vc & (1 << sender_channel_id)) != 0;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_receiver_channel_data_forwarded(
    const FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r, size_t receiver_channel_id) {
    return (r.receiver_channel_data_forwarded_bitfield_by_vc & (1 << receiver_channel_id)) != 0;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_noc_send_type_used(
    const FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r,
    size_t vc_id,
    NocSendType noc_send_type) {
    return (r.used_noc_send_type_by_vc_bitfield[vc_id] & (1 << static_cast<uint8_t>(noc_send_type))) != 0;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_sender_channel_forwarded_to(
    const FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>& r,
    size_t vc_id,
    size_t receiver_channel_id) {
    return (r.sender_channel_forwarded_to_bitfield_by_vc[vc_id] & (1 << receiver_channel_id)) != 0;
}

// ============================================================================
// Overloads for FabricDatapathUsageL1Results<false> (disabled) — all no-ops
// ============================================================================

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_reset(FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_sender_channel_used(
    FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_receiver_channel_data_forwarded(
    FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_update_sender_channel_packet_size(
    FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t, uint16_t) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_sender_channel_forwarded_to(
    FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t, size_t) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_merge_sender_channel_forwarded_to(
    FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t, uint16_t) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE void datapath_usage_set_noc_send_type_used(
    FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t, NocSendType) {}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_sender_channel_used(
    const FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t) {
    return false;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_receiver_channel_data_forwarded(
    const FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t) {
    return false;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_noc_send_type_used(
    const FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t, NocSendType) {
    return false;
}

template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
FORCE_INLINE bool datapath_usage_is_sender_channel_forwarded_to(
    const FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS>&, size_t, size_t) {
    return false;
}

// ============================================================================
// FabricDatapathUsageL1Ptr — compile-time L1 pointer wrapper
// Provides the same call-site API as before, delegating to the free functions.
// ============================================================================

// Primary template - enabled implementation
template <bool ENABLED, size_t L1_ADDR, size_t NUM_VC = 2, size_t MAX_NUM_SENDER_CHANNELS = 9>
struct FabricDatapathUsageL1Ptr {
    using ResultsType = FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>;
    FORCE_INLINE ResultsType* get() const { return reinterpret_cast<ResultsType*>(L1_ADDR); }

    FORCE_INLINE void reset() const { datapath_usage_reset(*get()); }
    FORCE_INLINE void set_sender_channel_used(size_t id) const { datapath_usage_set_sender_channel_used(*get(), id); }
    FORCE_INLINE void set_receiver_channel_data_forwarded(size_t id) const {
        datapath_usage_set_receiver_channel_data_forwarded(*get(), id);
    }
    FORCE_INLINE void update_sender_channel_packet_size(size_t id, uint16_t sz) const {
        datapath_usage_update_sender_channel_packet_size(*get(), id, sz);
    }
    FORCE_INLINE void set_sender_channel_forwarded_to(size_t vc, size_t rx) const {
        datapath_usage_set_sender_channel_forwarded_to(*get(), vc, rx);
    }
    FORCE_INLINE void merge_sender_channel_forwarded_to(size_t vc, uint16_t mask) const {
        datapath_usage_merge_sender_channel_forwarded_to(*get(), vc, mask);
    }
    FORCE_INLINE void set_noc_send_type_used(size_t vc, NocSendType t) const {
        datapath_usage_set_noc_send_type_used(*get(), vc, t);
    }
};

// Disabled specialization - all no-ops
template <size_t L1_ADDR, size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
struct FabricDatapathUsageL1Ptr<false, L1_ADDR, NUM_VC, MAX_NUM_SENDER_CHANNELS> {
    FORCE_INLINE void reset() const {}
    FORCE_INLINE void set_sender_channel_used(size_t) const {}
    FORCE_INLINE void set_receiver_channel_data_forwarded(size_t) const {}
    FORCE_INLINE void update_sender_channel_packet_size(size_t, uint16_t) const {}
    FORCE_INLINE void set_sender_channel_forwarded_to(size_t, size_t) const {}
    FORCE_INLINE void merge_sender_channel_forwarded_to(size_t, uint16_t) const {}
    FORCE_INLINE void set_noc_send_type_used(size_t, NocSendType) const {}
};

}  // namespace tt::tt_fabric
