// RMM HIP stub - device_buffer.hpp
// Minimal implementation for HIP builds
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace rmm {
namespace detail {
inline void check_hip_error(hipError_t err, const char* msg);  // Defined in device_uvector.hpp
} // namespace detail
} // namespace rmm

namespace rmm {

class device_buffer {
public:
    device_buffer() : data_(nullptr), size_(0), stream_(0) {}

    device_buffer(std::size_t size, hipStream_t stream = 0,
                  mr::device_memory_resource* mr = nullptr)
        : size_(size), stream_(stream) {
        if (size > 0) {
            hipError_t err = hipMalloc(&data_, size);
            if (err != hipSuccess) {
                throw std::runtime_error(std::string("device_buffer allocation failed: ") + hipGetErrorString(err));
            }
        } else {
            data_ = nullptr;
        }
    }

    ~device_buffer() {
        if (data_) {
            (void)hipFree(data_);  // Ignore error in destructor
        }
    }

    device_buffer(device_buffer&& other) noexcept
        : data_(other.data_), size_(other.size_), stream_(other.stream_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    device_buffer& operator=(device_buffer&& other) noexcept {
        if (this != &other) {
            if (data_) (void)hipFree(data_);  // Ignore error in move
            data_ = other.data_;
            size_ = other.size_;
            stream_ = other.stream_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    std::size_t size() const { return size_; }
    hipStream_t stream() const { return stream_; }

    void resize(std::size_t new_size, hipStream_t stream = 0) {
        if (new_size == size_) return;
        void* new_data = nullptr;
        if (new_size > 0) {
            hipError_t err = hipMalloc(&new_data, new_size);
            if (err != hipSuccess) {
                throw std::runtime_error(std::string("device_buffer resize failed: ") + hipGetErrorString(err));
            }
            if (data_ && size_ > 0) {
                err = hipMemcpyAsync(new_data, data_, std::min(size_, new_size),
                              hipMemcpyDeviceToDevice, stream);
                if (err != hipSuccess) {
                    (void)hipFree(new_data);
                    throw std::runtime_error(std::string("device_buffer resize copy failed: ") + hipGetErrorString(err));
                }
            }
        }
        if (data_) (void)hipFree(data_);
        data_ = new_data;
        size_ = new_size;
    }

private:
    void* data_;
    std::size_t size_;
    hipStream_t stream_;
};

} // namespace rmm
