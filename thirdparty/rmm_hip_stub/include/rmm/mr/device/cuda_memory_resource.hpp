// RMM HIP stub - cuda_memory_resource.hpp
// Maps to HIP memory operations
#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace rmm {
namespace mr {

class cuda_memory_resource : public device_memory_resource {
public:
    cuda_memory_resource() = default;
    ~cuda_memory_resource() override = default;

protected:
    void* do_allocate(std::size_t bytes, hipStream_t /*stream*/) override {
        void* ptr = nullptr;
        if (bytes > 0) {
            hipError_t err = hipMalloc(&ptr, bytes);
            if (err != hipSuccess) {
                throw std::runtime_error(std::string("cuda_memory_resource allocation failed: ") + hipGetErrorString(err));
            }
        }
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t /*bytes*/, hipStream_t /*stream*/) override {
        if (ptr) {
            (void)hipFree(ptr);  // Ignore error in deallocate
        }
    }
};

} // namespace mr
} // namespace rmm
