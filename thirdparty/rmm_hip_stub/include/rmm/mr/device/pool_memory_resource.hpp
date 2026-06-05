// RMM HIP stub - pool_memory_resource.hpp
// Simplified pool that just wraps the upstream resource (no actual pooling for stub)
#pragma once

#include <rmm/mr/device/device_memory_resource.hpp>
#include <memory>

namespace rmm {
namespace mr {

template <typename Upstream>
class pool_memory_resource : public device_memory_resource {
public:
    explicit pool_memory_resource(Upstream* upstream,
                                  std::size_t initial_pool_size = 0,
                                  std::size_t maximum_pool_size = 0)
        : upstream_(upstream), pool_size_(maximum_pool_size > 0 ? maximum_pool_size : initial_pool_size) {}

    explicit pool_memory_resource(std::shared_ptr<Upstream> upstream,
                                  std::size_t initial_pool_size = 0,
                                  std::size_t maximum_pool_size = 0)
        : upstream_shared_(upstream), upstream_(upstream.get()),
          pool_size_(maximum_pool_size > 0 ? maximum_pool_size : initial_pool_size) {}

    ~pool_memory_resource() override = default;

    Upstream* get_upstream() const { return upstream_; }

    // Return the configured pool size (stub doesn't actually pre-allocate)
    std::size_t pool_size() const { return pool_size_; }

protected:
    void* do_allocate(std::size_t bytes, hipStream_t stream) override {
        return upstream_->allocate(bytes, stream);
    }

    void do_deallocate(void* ptr, std::size_t bytes, hipStream_t stream) override {
        upstream_->deallocate(ptr, bytes, stream);
    }

private:
    std::shared_ptr<Upstream> upstream_shared_;
    Upstream* upstream_;
    std::size_t pool_size_;
};

} // namespace mr
} // namespace rmm
