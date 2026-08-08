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
# RMM depends on CUDA. For HIP builds, we use a simple stub implementation.
if(NOT USE_HIP)
    FetchContent_Declare(
        rmm
        GIT_REPOSITORY https://github.com/rapidsai/rmm
        GIT_TAG        branch-25.08
        SOURCE_SUBDIR  cpp
    )
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(rmm)
else()
    # For HIP builds, create a stub rmm target that provides minimal compatibility
    add_library(rmm INTERFACE)
    target_include_directories(rmm INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rmm_hip_stub/include>
        $<INSTALL_INTERFACE:include/rmm_hip_stub>)
    # Use hip::host (not hip::device) to avoid propagating HIP compile flags
    # to pure C++ consumers like test executables compiled with g++.
    target_link_libraries(rmm INTERFACE hip::host)
    # Install the interface library so exports work
    install(TARGETS rmm EXPORT rmm_targets)
    install(EXPORT rmm_targets DESTINATION lib/cmake/rmm)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rmm_hip_stub/include/ DESTINATION include/rmm_hip_stub)
endif()

# The submodules include <cuda_runtime.h> and <curand_kernel.h> by name. Putting
# hip_compat/ ahead of them on the include path resolves those to HIP shims and
# leaves the submodule sources' include lines untouched.
if(USE_HIP)
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
endif()

# Pass USE_HIP to subdirectories
add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

