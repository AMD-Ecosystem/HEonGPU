# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

if(USE_HIP)
    set(THIRDPARTY_BUILD_ARG ON)
else()
    set(THIRDPARTY_BUILD_ARG OFF)
endif()

execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh ${THIRDPARTY_BUILD_ARG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE BUILD_SCRIPT_RESULT
    OUTPUT_VARIABLE BUILD_SCRIPT_OUTPUT
    ERROR_VARIABLE BUILD_SCRIPT_ERROR
)

if(NOT BUILD_SCRIPT_RESULT EQUAL 0)
    message(FATAL_ERROR "Thirdparty build script failed:\n${BUILD_SCRIPT_ERROR}")
endif()

message(STATUS "Thirdparty build completed successfully")

include(FetchContent)

# RMM (RAPIDS Memory Manager)
# On HIP builds the same interface comes from hipMM, the ROCm-DS port of RMM,
# which keeps RMM's target name, rmm:: namespace, and header layout, so the
# sources that use rmm compile unchanged on both platforms.
if(NOT USE_HIP)
    FetchContent_Declare(
        rmm
        GIT_REPOSITORY https://github.com/rapidsai/rmm
        GIT_TAG        branch-25.08
        SOURCE_SUBDIR  cpp
    )
else()
    FetchContent_Declare(
        rmm
        GIT_REPOSITORY https://github.com/AMD-Ecosystem/hipMM
        # release/rocmds-26.03 line, pinned by commit: hipMM publishes no
        # release tags yet and the branch moves.
        GIT_TAG        22732e49aa00ba179847e3b314371f2307f47a9a
        SOURCE_SUBDIR  cpp
    )
endif()
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(rmm)
if(USE_HIP)
    # hipMM expects a HIP-capable compiler for its own sources (its hip
    # dependency injects -x hip into their compile flags, which a plain C++
    # compiler rejects). Compiling them as HIP sources satisfies that the same
    # way this port compiles its own GPU sources.
    get_target_property(_rmm_source_dir rmm SOURCE_DIR)
    get_target_property(_rmm_sources rmm SOURCES)
    list(TRANSFORM _rmm_sources PREPEND "${_rmm_source_dir}/")
    set_source_files_properties(${_rmm_sources} TARGET_DIRECTORY rmm
                                PROPERTIES LANGUAGE HIP)
endif()

# The submodules include <cuda_runtime.h> and <curand_kernel.h> by name. Putting
# hip_compat/ ahead of them on the include path resolves those to HIP shims and
# leaves the submodule sources' include lines untouched.
if(USE_HIP)
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat/ DESTINATION include/hip_compat)
endif()

# Pass USE_HIP to subdirectories
add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

