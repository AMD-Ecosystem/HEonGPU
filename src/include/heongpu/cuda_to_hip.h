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

#else // CUDA path

// Deliberately no <curand_kernel.h> here. curand_mtgp32_kernel.h declares
// threadIdx and blockDim with C++ linkage and device_launch_parameters.h
// declares the same names with C linkage, and a host compiler rejects the pair
// only in one order: curand first. Host translation units reach this header
// first of all (util.cuh includes it ahead of everything else), and the
// <cuda_runtime.h> below does not put device_launch_parameters.h in front,
// because cuda_runtime.h includes it only under __CUDACC__. A curand include
// here therefore always arrived first and broke every host .cpp. The kernel
// headers that use curand include it themselves and are reached later, after
// device_launch_parameters.h, which is the order that compiles.
#include <cuda_runtime.h>

#endif // USE_HIP
