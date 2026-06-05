// RMM HIP stub - host_memory_resource.hpp
#pragma once

#include <cstddef>
#include <cstdlib>

namespace rmm {
namespace mr {

class host_memory_resource {
public:
    virtual ~host_memory_resource() = default;

    void* allocate(std::size_t bytes) {
        return do_allocate(bytes);
    }

    void deallocate(void* ptr, std::size_t bytes) {
        do_deallocate(ptr, bytes);
    }

protected:
    virtual void* do_allocate(std::size_t bytes) = 0;
    virtual void do_deallocate(void* ptr, std::size_t bytes) = 0;
};

} // namespace mr
} // namespace rmm
