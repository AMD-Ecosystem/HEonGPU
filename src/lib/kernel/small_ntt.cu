// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Implementations are now inline in the header (small_ntt.cuh) to avoid
// cross-translation-unit device linking issues on HIP.
// This file is kept for CUDA builds that may still use explicit instantiation.
#include <heongpu/kernel/small_ntt.cuh>