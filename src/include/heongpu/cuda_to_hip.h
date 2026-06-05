// CUDA to HIP compatibility header for HEonGPU
// This file provides CUDA->HIP symbol aliases when building with ROCm/HIP.
// On CUDA, it simply includes cuda_runtime.h and does nothing else.
//
// This port was authored with an AI assistant (Claude).

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// CUDA runtime -> HIP runtime
#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

#define cudaStream_t                hipStream_t
#define cudaStreamDefault           hipStreamDefault
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize

#define cudaEvent_t                 hipEvent_t
#define cudaEventCreate             hipEventCreate
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventDestroy            hipEventDestroy
#define cudaEventElapsedTime        hipEventElapsedTime

#define cudaError_t                 hipError_t
#define cudaSuccess                 hipSuccess
#define cudaGetErrorString          hipGetErrorString
#define cudaGetLastError            hipGetLastError
#define cudaPeekAtLastError         hipPeekAtLastError
#define cudaDeviceSynchronize       hipDeviceSynchronize

#define cudaGetDevice               hipGetDevice
#define cudaSetDevice               hipSetDevice
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceProp              hipDeviceProp_t
#define cudaMemGetInfo              hipMemGetInfo

#define cudaFuncSetAttribute        hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

// Device limits
#define cudaDeviceSetLimit          hipDeviceSetLimit
#define cudaLimitStackSize          hipLimitStackSize

// cuRAND device API -> hipRAND device API
#define curandState                 hiprandState
#define curandState_t               hiprandState_t
#define curandStateXORWOW           hiprandStateXORWOW
#define curandStateXORWOW_t         hiprandStateXORWOW_t
#define curand_init                 hiprand_init
#define curand                      hiprand
#define curand_uniform              hiprand_uniform
#define curand_uniform_double       hiprand_uniform_double
#define curand_normal               hiprand_normal
#define curand_normal_double        hiprand_normal_double

// Warp size abstraction for device code
// On AMD CDNA (gfx9xx), wavefront is 64 lanes; on RDNA (gfx10xx/gfx11xx), it is 32.
// Use kWarpSize for compile-time device code; use warpSize for runtime (correct on both).
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
static constexpr int kWarpSize = 64;   // CDNA: gfx90a, gfx94x
#else
static constexpr int kWarpSize = 32;   // RDNA: gfx10xx, gfx11xx
#endif
// For shuffle masks on HIP, we need 64-bit masks regardless of actual wavefront width
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#else
// Host code: kWarpSize not used, but define for compatibility
static constexpr int kWarpSize = 64;   // Upper bound for host-side sizing
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#endif

#else // CUDA path

#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA uses 32-lane warps everywhere
#if defined(__CUDA_ARCH__)
static constexpr int kWarpSize = 32;
#else
static constexpr int kWarpSize = 32;
#endif
#define FULL_WARP_MASK 0xFFFFFFFF

#endif // USE_HIP
