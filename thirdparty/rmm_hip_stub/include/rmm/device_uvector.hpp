// RMM HIP stub - device_uvector.hpp
// Minimal implementation for HIP builds
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace rmm {
namespace detail {
inline void check_hip_error(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}
} // namespace detail
} // namespace rmm

namespace rmm {

template <typename T>
class device_uvector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;

    device_uvector() : data_(nullptr), size_(0), capacity_(0), stream_(0) {}

    explicit device_uvector(std::size_t size, hipStream_t stream = 0,
                           mr::device_memory_resource* mr = nullptr)
        : size_(size), capacity_(size), stream_(stream) {
        if (size > 0) {
            detail::check_hip_error(hipMalloc(&data_, size * sizeof(T)), "device_uvector allocation failed");
        } else {
            data_ = nullptr;
        }
    }

    device_uvector(const std::vector<T>& host_vec, hipStream_t stream = 0,
                   mr::device_memory_resource* mr = nullptr)
        : size_(host_vec.size()), capacity_(host_vec.size()), stream_(stream) {
        if (size_ > 0) {
            detail::check_hip_error(hipMalloc(&data_, size_ * sizeof(T)), "device_uvector allocation failed");
            detail::check_hip_error(hipMemcpyAsync(data_, host_vec.data(), size_ * sizeof(T),
                          hipMemcpyHostToDevice, stream), "device_uvector copy failed");
        } else {
            data_ = nullptr;
        }
    }

    ~device_uvector() {
        if (data_) {
            (void)hipFree(data_);  // Ignore error in destructor
        }
    }

    device_uvector(device_uvector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
          stream_(other.stream_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    device_uvector& operator=(device_uvector&& other) noexcept {
        if (this != &other) {
            if (data_) (void)hipFree(data_);  // Ignore error in move
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            stream_ = other.stream_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    device_uvector(const device_uvector&) = delete;
    device_uvector& operator=(const device_uvector&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    T* begin() { return data_; }
    const T* begin() const { return data_; }
    T* end() { return data_ + size_; }
    const T* end() const { return data_ + size_; }
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    hipStream_t stream() const { return stream_; }

    void resize(std::size_t new_size, hipStream_t stream = 0) {
        if (new_size <= capacity_) {
            size_ = new_size;
            return;
        }
        T* new_data = nullptr;
        if (new_size > 0) {
            detail::check_hip_error(hipMalloc(&new_data, new_size * sizeof(T)), "device_uvector resize failed");
            if (data_ && size_ > 0) {
                detail::check_hip_error(hipMemcpyAsync(new_data, data_, size_ * sizeof(T),
                              hipMemcpyDeviceToDevice, stream), "device_uvector resize copy failed");
            }
        }
        if (data_) (void)hipFree(data_);
        data_ = new_data;
        size_ = new_size;
        capacity_ = new_size;
    }

    void reserve(std::size_t new_capacity, hipStream_t stream = 0) {
        if (new_capacity <= capacity_) return;
        T* new_data = nullptr;
        detail::check_hip_error(hipMalloc(&new_data, new_capacity * sizeof(T)), "device_uvector reserve failed");
        if (data_ && size_ > 0) {
            detail::check_hip_error(hipMemcpyAsync(new_data, data_, size_ * sizeof(T),
                          hipMemcpyDeviceToDevice, stream), "device_uvector reserve copy failed");
        }
        if (data_) (void)hipFree(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }

    void clear() { size_ = 0; }

    void shrink_to_fit(hipStream_t stream = 0) {
        if (capacity_ > size_) {
            T* new_data = nullptr;
            if (size_ > 0) {
                detail::check_hip_error(hipMalloc(&new_data, size_ * sizeof(T)), "device_uvector shrink_to_fit failed");
                detail::check_hip_error(hipMemcpyAsync(new_data, data_, size_ * sizeof(T),
                              hipMemcpyDeviceToDevice, stream), "device_uvector shrink_to_fit copy failed");
            }
            if (data_) (void)hipFree(data_);
            data_ = new_data;
            capacity_ = size_;
        }
    }

    void set_stream(hipStream_t stream) { stream_ = stream; }

private:
    T* data_;
    std::size_t size_;
    std::size_t capacity_;
    hipStream_t stream_;
};

} // namespace rmm
