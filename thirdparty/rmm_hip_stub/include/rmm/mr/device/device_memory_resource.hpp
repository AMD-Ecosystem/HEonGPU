// RMM HIP stub - device_memory_resource.hpp
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

namespace rmm {
namespace mr {

class device_memory_resource {
public:
    virtual ~device_memory_resource() = default;

    void* allocate(std::size_t bytes, hipStream_t stream = 0) {
        return do_allocate(bytes, stream);
    }

    void deallocate(void* ptr, std::size_t bytes, hipStream_t stream = 0) {
        do_deallocate(ptr, bytes, stream);
    }

    bool is_equal(const device_memory_resource& other) const noexcept {
        return do_is_equal(other);
    }

protected:
    virtual void* do_allocate(std::size_t bytes, hipStream_t stream) = 0;
    virtual void do_deallocate(void* ptr, std::size_t bytes, hipStream_t stream) = 0;
    virtual bool do_is_equal(const device_memory_resource& other) const noexcept {
        return this == &other;
    }
};

// Simple global device memory resource management
namespace detail {
    inline device_memory_resource*& get_current_device_resource_ref() {
        static device_memory_resource* current_mr = nullptr;
        return current_mr;
    }
}

inline device_memory_resource* get_current_device_resource() {
    return detail::get_current_device_resource_ref();
}

inline device_memory_resource* set_current_device_resource(device_memory_resource* mr) {
    auto old = detail::get_current_device_resource_ref();
    detail::get_current_device_resource_ref() = mr;
    return old;
}

} // namespace mr
} // namespace rmm
