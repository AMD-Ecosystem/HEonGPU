// RMM HIP stub - pinned_memory_resource.hpp
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rmm {
namespace mr {

class pinned_memory_resource {
public:
    pinned_memory_resource() = default;
    ~pinned_memory_resource() = default;

    // Single-argument version (for direct calls)
    void* allocate(std::size_t bytes) {
        void* ptr = nullptr;
        if (bytes > 0) {
            hipError_t err = hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
            if (err != hipSuccess) {
                throw std::runtime_error(std::string("pinned_memory_resource allocation failed: ") + hipGetErrorString(err));
            }
        }
        return ptr;
    }

    // Two-argument version (for compatibility with pool_memory_resource template)
    void* allocate(std::size_t bytes, hipStream_t /*stream*/) {
        return allocate(bytes);
    }

    void deallocate(void* ptr, std::size_t /*bytes*/) {
        if (ptr) {
            (void)hipHostFree(ptr);  // Ignore error in deallocate
        }
    }

    void deallocate(void* ptr, std::size_t bytes, hipStream_t /*stream*/) {
        deallocate(ptr, bytes);
    }
};

} // namespace mr
} // namespace rmm
