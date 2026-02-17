// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "core_config.h"
#include "dev_mem_map.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type = DebugAssertTripped) {
    // Write the line number into the memory mailbox for host to read.
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->tripped = assert_type;
        std::uint64_t cpu_index = 0;
#if defined(ARCH_QUASAR)
        // TODO: The below code is recurring on Quasar.
        // It needs to be in a get_cpu_idx() API for Quasar
#if defined(COMPILE_FOR_TRISC)
        std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
        std::uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
        cpu_index = MaxDMProcessorsPerCoreType + NUM_TRISC_CORES * neo_id + trisc_id;  // after 8 DM cores
#else
        asm volatile("csrr %0, mhartid" : "=r"(cpu_index));
#endif
#else
        cpu_index = PROCESSOR_INDEX;
#endif
        v->which = cpu_index;
    }

    // Hang, or in the case of erisc, early exit.
#if defined(COMPILE_FOR_ERISC)
    // Update launch msg to show that we've exited. This is required so that the next run doesn't think there's a kernel
    // still running and try to make it exit.
    volatile tt_l1_ptr go_msg_t* go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_ptr->signal = RUN_MSG_DONE;

    // This exits to base FW
    internal_::disable_erisc_app();
    // Subordinates do not have an erisc exit
#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
    erisc_exit();
#endif
#endif

    while (1) {
        ;
    }
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define ASSERT(condition, ...)                        \
    do {                                              \
        if (not(condition))                           \
            assert_and_hang(__LINE__, ##__VA_ARGS__); \
    } while (0)

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#else  // !WATCHER_ENABLED

#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)

#define ASSERT(condition, ...)      \
    do {                            \
        if (!(condition))           \
            asm volatile("ebreak"); \
    } while (0)

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#else  // !LIGHTWEIGHT_KERNEL_ASSERTS

#define ASSERT(condition, ...)

#define ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#endif  // !LIGHTWEIGHT_KERNEL_ASSERTS

#endif  // WATCHER_ENABLED
