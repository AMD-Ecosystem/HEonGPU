// HIP compatibility shim for cuda_runtime.h
// This header redirects to hip_runtime.h when building with HIP.

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

// CUDA runtime -> HIP runtime symbol aliases
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

// Pointer attributes
#define cudaPointerAttributes       hipPointerAttribute_t
#define cudaPointerGetAttributes    hipPointerGetAttributes

// Memory types
#define cudaMemoryTypeDevice        hipMemoryTypeDevice
#define cudaMemoryTypeHost          hipMemoryTypeHost
#define cudaMemoryTypeUnregistered  hipMemoryTypeUnregistered
#define cudaMemoryTypeManaged       hipMemoryTypeManaged

// Device attributes
#define cudaDeviceGetAttribute      hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

// Async memory allocation
#define cudaMallocAsync             hipMallocAsync
#define cudaFreeAsync               hipFreeAsync

#else
#include_next <cuda_runtime.h>
#endif
