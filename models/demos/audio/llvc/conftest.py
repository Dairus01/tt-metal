# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local pytest config for the LLVC demo tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_device: test requires a Tenstorrent device and is skipped in CPU-only CI",
    )
