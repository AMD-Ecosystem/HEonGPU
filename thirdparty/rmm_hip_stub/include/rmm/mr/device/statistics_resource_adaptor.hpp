// RMM HIP stub - statistics_resource_adaptor.hpp
#pragma once

#include <rmm/mr/device/device_memory_resource.hpp>
#include <memory>
#include <atomic>

namespace rmm {
namespace mr {

// Counter struct to match real RMM API
struct counter {
    std::size_t value{0};
    std::size_t peak{0};
    counter() = default;
    counter(std::size_t v, std::size_t p) : value(v), peak(p) {}
};

template <typename Upstream>
class statistics_resource_adaptor : public device_memory_resource {
public:
    explicit statistics_resource_adaptor(Upstream* upstream)
        : upstream_(upstream), bytes_allocated_(0), bytes_deallocated_(0) {}

    explicit statistics_resource_adaptor(std::shared_ptr<Upstream> upstream)
        : upstream_shared_(upstream), upstream_(upstream.get()),
          bytes_allocated_(0), bytes_deallocated_(0) {}

    ~statistics_resource_adaptor() override = default;

    Upstream* get_upstream() const { return upstream_; }

    // Returns counter struct with .value and .peak members
    counter get_bytes_counter() const {
        std::size_t current = bytes_allocated_ - bytes_deallocated_;
        return counter{current, peak_bytes_};
    }
    counter get_allocations_counter() const { return counter{allocations_, peak_allocations_}; }

protected:
    void* do_allocate(std::size_t bytes, hipStream_t stream) override {
        void* ptr = upstream_->allocate(bytes, stream);
        bytes_allocated_ += bytes;
        ++allocations_;
        // Track peak
        std::size_t current = bytes_allocated_ - bytes_deallocated_;
        std::size_t peak = peak_bytes_.load();
        while (current > peak && !peak_bytes_.compare_exchange_weak(peak, current)) {}
        std::size_t allocs = allocations_.load();
        std::size_t peak_allocs = peak_allocations_.load();
        while (allocs > peak_allocs && !peak_allocations_.compare_exchange_weak(peak_allocs, allocs)) {}
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t bytes, hipStream_t stream) override {
        upstream_->deallocate(ptr, bytes, stream);
        bytes_deallocated_ += bytes;
    }

private:
    std::shared_ptr<Upstream> upstream_shared_;
    Upstream* upstream_;
    std::atomic<std::size_t> bytes_allocated_;
    std::atomic<std::size_t> bytes_deallocated_;
    std::atomic<std::size_t> allocations_{0};
    std::atomic<std::size_t> peak_bytes_{0};
    std::atomic<std::size_t> peak_allocations_{0};
};

} // namespace mr
} // namespace rmm
