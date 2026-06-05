// RMM HIP stub - logging_resource_adaptor.hpp
#pragma once

#include <rmm/mr/device/device_memory_resource.hpp>
#include <memory>

namespace rmm {
namespace mr {

template <typename Upstream>
class logging_resource_adaptor : public device_memory_resource {
public:
    explicit logging_resource_adaptor(Upstream* upstream)
        : upstream_(upstream) {}

    explicit logging_resource_adaptor(std::shared_ptr<Upstream> upstream)
        : upstream_shared_(upstream), upstream_(upstream.get()) {}

    ~logging_resource_adaptor() override = default;

    Upstream* get_upstream() const { return upstream_; }

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
};

} // namespace mr
} // namespace rmm
