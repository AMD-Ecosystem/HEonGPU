// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifdef _WIN32
// Must precede any other header that may pull in <windows.h>, so that the
// min/max macros never reach the C++ standard library or Thrust.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <heongpu/util/memorypool.cuh>

namespace
{
    // hipMM, which provides RMM on AMD GPUs, follows the RMM 25.x API: it
    // allocates with allocate_async(bytes, alignment, stream) and sets the
    // default resource with set_current_device_resource_ref.
#if defined(__HIP_PLATFORM_AMD__)
    template <typename Resource>
    void* rmm_allocate(Resource& resource, cudaStream_t stream, size_t size)
    {
        return resource.allocate_async(size, rmm::CUDA_ALLOCATION_ALIGNMENT,
                                       stream);
    }

    template <typename Resource>
    void rmm_deallocate(Resource& resource, cudaStream_t stream, void* ptr,
                        size_t size)
    {
        resource.deallocate_async(ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT,
                                  stream);
    }

    template <typename Resource> auto set_rmm_default(Resource&& resource)
    {
        return rmm::mr::set_current_device_resource_ref(
            std::forward<Resource>(resource));
    }
#else
    template <typename Resource>
    void* rmm_allocate(Resource& resource, cudaStream_t stream, size_t size)
    {
        return resource.allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    template <typename Resource>
    void rmm_deallocate(Resource& resource, cudaStream_t stream, void* ptr,
                        size_t size)
    {
        resource.deallocate(stream, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    template <typename Resource> auto set_rmm_default(Resource&& resource)
    {
        return rmm::mr::set_current_device_resource(
            std::forward<Resource>(resource));
    }
#endif
} // namespace

namespace heongpu
{
    std::shared_ptr<MemoryPool::HostResource> MemoryPool::host_base_ = nullptr;
    std::shared_ptr<MemoryPool::HostPoolResource> MemoryPool::host_pool_ =
        nullptr;
    std::shared_ptr<MemoryPool::HostStatsAdaptor>
        MemoryPool::host_stats_adaptor_ = nullptr;

    std::shared_ptr<MemoryPool::DeviceResource> MemoryPool::device_base_ =
        nullptr;
    std::shared_ptr<MemoryPool::DevicePoolResource> MemoryPool::device_pool_ =
        nullptr;
    std::shared_ptr<MemoryPool::DeviceStatsAdaptor>
        MemoryPool::device_stats_adaptor_ = nullptr;
    bool MemoryPool::initialized_ = false;
    bool MemoryPool::use_pool_ = true;
    std::unordered_set<void*> MemoryPool::pooled_device_allocations_;
    std::unordered_set<void*> MemoryPool::pooled_host_allocations_;
    std::optional<MemoryPool::RmmDefaultResource>
        MemoryPool::previous_rmm_default_;
    std::mutex MemoryPool::mutex_;

    MemoryPoolConfig MemoryPoolConfig::Defaults()
    {
        MemoryPoolConfig config;
        config.initial_device_fraction = initial_device_memorypool_size;
        config.max_device_fraction = max_device_memorypool_size;
        config.max_host_fraction = max_host_memorypool_size;
        return config;
    }

    MemoryPool& MemoryPool::instance()
    {
        // Initialize CUDA runtime before the singleton is constructed so that
        // teardown order stays correct at process exit.
        cudaFree(nullptr);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        static MemoryPool instance;
        return instance;
    }

    void MemoryPool::ensure_base_resources()
    {
        if (!host_base_)
        {
            host_base_ = std::make_shared<HostResource>();
        }
        if (!device_base_)
        {
            device_base_ = std::make_shared<DeviceResource>();
        }
    }

    size_t MemoryPool::get_host_avaliable_memory() const
    {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo{};
        memInfo.dwLength = sizeof(memInfo);
        GlobalMemoryStatusEx(&memInfo);
        size_t free_memory = static_cast<size_t>(memInfo.ullAvailPhys);
        return free_memory;
#else
        struct sysinfo memInfo;
        sysinfo(&memInfo);
        size_t free_memory = memInfo.freeram * memInfo.mem_unit;
        return free_memory;
#endif
    }

    size_t MemoryPool::get_decive_avaliable_memory() const
    {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        return free_mem; // total_mem
    }

    size_t MemoryPool::roundup_256(size_t size) const
    {
        return ((size + 255) / 256) * 256;
    }

    void MemoryPool::initialize()
    {
        initialize(MemoryPoolConfig::Defaults());
    }

    void MemoryPool::initialize(const MemoryPoolConfig& config)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        // The pools are created by the first initialization that requests
        // them, even if an earlier one ran without a pool.
        if (!initialized_ || (config.use_memory_pool && !device_pool_))
        {
            ensure_base_resources();
            size_t total_host_memory = get_host_avaliable_memory();
            size_t total_device_memory = get_decive_avaliable_memory();

            auto normalize_fraction = [](float value,
                                         const char* label) -> float
            {
                if (value <= 0.0f)
                {
                    throw std::invalid_argument(std::string(label) +
                                                " must be > 0");
                }
                if (value > 1.0f)
                {
                    if (value > 100.0f)
                    {
                        throw std::invalid_argument(
                            std::string(label) +
                            " must be in (0,1] or (0,100]");
                    }
                    return value / 100.0f;
                }
                return value;
            };

            auto resolve_pool_size =
                [&](const std::optional<size_t>& bytes,
                    const std::optional<float>& fraction, size_t total_memory,
                    size_t default_bytes, float default_fraction,
                    const char* label) -> size_t
            {
                size_t resolved = 0;
                if (bytes.has_value())
                {
                    if (*bytes == 0)
                    {
                        throw std::invalid_argument(std::string(label) +
                                                    " bytes must be > 0");
                    }
                    resolved = *bytes;
                }
                else if (fraction.has_value())
                {
                    float f = normalize_fraction(*fraction, label);
                    resolved = static_cast<size_t>(total_memory * f);
                }
                else if (default_bytes != 0)
                {
                    resolved = default_bytes;
                }
                else
                {
                    float f = normalize_fraction(default_fraction, label);
                    resolved = static_cast<size_t>(total_memory * f);
                }

                if (resolved == 0)
                {
                    throw std::invalid_argument(std::string(label) +
                                                " resolved to 0 bytes");
                }
                if (total_memory != 0 && resolved > total_memory)
                {
                    throw std::invalid_argument(
                        std::string(label) +
                        " exceeds available memory at initialization");
                }

                return roundup_256(resolved);
            };

            if (config.use_memory_pool)
            {
                size_t initial_host_pool_size = 0;
                if (!config.initial_host_bytes.has_value() &&
                    !config.initial_host_fraction.has_value())
                {
                    initial_host_pool_size =
                        roundup_256(static_cast<size_t>(104857600));
                }
                else
                {
                    initial_host_pool_size = resolve_pool_size(
                        config.initial_host_bytes, config.initial_host_fraction,
                        total_host_memory, 0, initial_host_memorypool_size,
                        "host initial pool size");
                }

                size_t max_host_pool_size = resolve_pool_size(
                    config.max_host_bytes, config.max_host_fraction,
                    total_host_memory, 0, max_host_memorypool_size,
                    "host max pool size");

                if (max_host_pool_size < initial_host_pool_size)
                {
                    throw std::invalid_argument(
                        "host max pool size must be >= host initial pool "
                        "size");
                }

                host_pool_ = std::make_shared<HostPoolResource>(
                    *host_base_, initial_host_pool_size, max_host_pool_size);
                host_stats_adaptor_ =
                    std::make_shared<HostStatsAdaptor>(*host_pool_);

                size_t initial_device_pool_size = resolve_pool_size(
                    config.initial_device_bytes, config.initial_device_fraction,
                    total_device_memory, 0, initial_device_memorypool_size,
                    "device initial pool size");
                size_t max_device_pool_size = resolve_pool_size(
                    config.max_device_bytes, config.max_device_fraction,
                    total_device_memory, 0, max_device_memorypool_size,
                    "device max pool size");

                if (max_device_pool_size < initial_device_pool_size)
                {
                    throw std::invalid_argument(
                        "device max pool size must be >= device initial pool "
                        "size");
                }

                // An initial size derived from the free memory can become too
                // large if another process allocates in the meantime. Retry
                // with smaller sizes; the pool grows on demand up to its max.
                constexpr size_t min_initial_device_pool_size = 64ULL << 20;
                const bool shrinkable =
                    !config.initial_device_bytes.has_value();
                while (true)
                {
                    if (shrinkable)
                    {
                        // Skip sizes that no longer fit instead of attempting
                        // them, which avoids failed allocations and RMM
                        // error logs.
                        constexpr size_t headroom = 64ULL << 20;
                        const size_t free_memory =
                            get_decive_avaliable_memory();
                        while (initial_device_pool_size >
                                   min_initial_device_pool_size &&
                               initial_device_pool_size + headroom >
                                   free_memory)
                        {
                            initial_device_pool_size =
                                roundup_256(initial_device_pool_size / 2);
                        }
                    }
                    try
                    {
                        device_pool_ = std::make_shared<DevicePoolResource>(
                            *device_base_, initial_device_pool_size,
                            max_device_pool_size);
                        break;
                    }
                    catch (const rmm::out_of_memory&)
                    {
                        if (!shrinkable || initial_device_pool_size <=
                                               min_initial_device_pool_size)
                        {
                            throw;
                        }
                        initial_device_pool_size =
                            roundup_256(initial_device_pool_size / 2);
                    }
                }
                device_stats_adaptor_ =
                    std::make_shared<DeviceStatsAdaptor>(*device_pool_);
            }

            initialized_ = true;
        }
    }

    void MemoryPool::use_memory_pool(bool use)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        use_pool_ = use;
        const rmm::device_async_resource_ref resource = device_resource();
        if (previous_rmm_default_)
        {
            // Keep the RMM default on the resource HEonGPU now uses.
            set_rmm_default(resource);
        }
    }

    void MemoryPool::set_as_rmm_default(bool enable)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (enable)
        {
            auto previous =
                set_rmm_default(device_resource());
            // Keep the default from before HEonGPU first replaced it.
            if (!previous_rmm_default_)
            {
                previous_rmm_default_ = std::move(previous);
            }
        }
        else if (previous_rmm_default_)
        {
            set_rmm_default(
                std::move(*previous_rmm_default_));
            previous_rmm_default_.reset();
        }
    }

    void MemoryPool::release()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        const size_t device_used =
            device_stats_adaptor_
                ? static_cast<size_t>(
                      device_stats_adaptor_->get_bytes_counter().value)
                : 0;
        const size_t host_used =
            host_stats_adaptor_
                ? static_cast<size_t>(
                      host_stats_adaptor_->get_bytes_counter().value)
                : 0;
        if (device_used != 0 || host_used != 0)
        {
            throw StateError(error_message(
                "HEonGPU can not be shut down while its memory pool is in "
                "use (",
                device_used, " bytes of device memory and ", host_used,
                " bytes of host memory). Destroy all HEonGPU objects "
                "(contexts, keys, plaintexts, ciphertexts, encoders, "
                "encryptors, decryptors and operators) first."));
        }
        if (previous_rmm_default_)
        {
            set_rmm_default(
                std::move(*previous_rmm_default_));
            previous_rmm_default_.reset();
        }
        // The base resources are kept: memory that was allocated without the
        // pool can still be returned to them.
        device_stats_adaptor_.reset();
        device_pool_.reset();
        host_stats_adaptor_.reset();
        host_pool_.reset();
        pooled_device_allocations_.clear();
        pooled_host_allocations_.clear();
        initialized_ = false;
        use_pool_ = true;
    }

    rmm::device_async_resource_ref MemoryPool::device_resource()
    {
        if (use_pool_ && device_stats_adaptor_)
        {
            return rmm::device_async_resource_ref{*device_stats_adaptor_};
        }
        ensure_base_resources();
        return rmm::device_async_resource_ref{*device_base_};
    }

    void* MemoryPool::allocate(size_t size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (use_pool_ && device_stats_adaptor_)
        {
            void* ptr = rmm_allocate(*device_stats_adaptor_, stream, size);
            pooled_device_allocations_.insert(ptr);
            return ptr;
        }
        ensure_base_resources();
        return rmm_allocate(*device_base_, stream, size);
    }

    void MemoryPool::deallocate(void* ptr, size_t size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (pooled_device_allocations_.erase(ptr) != 0)
        {
            rmm_deallocate(*device_stats_adaptor_, stream, ptr, size);
            return;
        }
        ensure_base_resources();
        rmm_deallocate(*device_base_, stream, ptr, size);
    }

    rmm::device_async_resource_ref MemoryPool::get_device_resource() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return const_cast<MemoryPool*>(this)->device_resource();
    }

    MemoryPool::HostStatsAdaptor* MemoryPool::get_host_resource() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return host_stats_adaptor_.get();
    }

    void* MemoryPool::host_allocate(size_t size)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (use_pool_ && host_stats_adaptor_)
        {
            void* ptr = rmm_allocate(*host_stats_adaptor_, nullptr, size);
            pooled_host_allocations_.insert(ptr);
            return ptr;
        }
        ensure_base_resources();
        return rmm_allocate(*host_base_, nullptr, size);
    }

    void MemoryPool::host_deallocate(void* ptr, size_t size)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (pooled_host_allocations_.erase(ptr) != 0)
        {
            rmm_deallocate(*host_stats_adaptor_, nullptr, ptr, size);
            return;
        }
        ensure_base_resources();
        rmm_deallocate(*host_base_, nullptr, ptr, size);
    }

    void MemoryPool::print_memory_pool_status(std::ostream& os) const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (device_stats_adaptor_ && device_pool_ && host_stats_adaptor_ &&
            host_pool_)
        {
            auto device_status = device_stats_adaptor_->get_bytes_counter();
            const auto device_total = device_pool_->pool_size();
            const auto device_used = device_status.value;
            const auto device_free = device_total - device_used;

            auto host_status = host_stats_adaptor_->get_bytes_counter();
            const auto host_total = host_pool_->pool_size();
            const auto host_used = host_status.value;
            const auto host_free = host_total - host_used;

            os << "[HEonGPU] Memory Pool Status" << std::endl;
            os << "  Device Pool:" << std::endl;
            os << "    Total     : " << device_total << " bytes" << std::endl;
            os << "    Used      : " << device_used << " bytes" << std::endl;
            os << "    Free      : " << device_free << " bytes" << std::endl;
            os << "  Host Pool:" << std::endl;
            os << "    Total     : " << host_total << " bytes" << std::endl;
            os << "    Used      : " << host_used << " bytes" << std::endl;
            os << "    Free      : " << host_free << " bytes" << std::endl;
        }
        else
        {
            os << "[HEonGPU] Memory pool is not initialized or is disabled."
               << std::endl;
        }
    }

    size_t MemoryPool::get_current_device_pool_memory_usage() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!device_stats_adaptor_)
        {
            return 0;
        }
        auto device_status = device_stats_adaptor_->get_bytes_counter();
        return device_status.value;
    }

    size_t MemoryPool::get_free_device_pool_memory() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!device_stats_adaptor_ || !device_pool_)
        {
            return 0;
        }
        auto device_status = device_stats_adaptor_->get_bytes_counter();
        return device_pool_->pool_size() - device_status.value;
    }

    size_t MemoryPool::get_current_host_pool_memory_usage() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!host_stats_adaptor_)
        {
            return 0;
        }
        auto host_status = host_stats_adaptor_->get_bytes_counter();
        return host_status.value;
    }

    size_t MemoryPool::get_free_host_pool_memory() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!host_stats_adaptor_ || !host_pool_)
        {
            return 0;
        }
        auto host_status = host_stats_adaptor_->get_bytes_counter();
        return host_pool_->pool_size() - host_status.value;
    }

    MemoryPool::~MemoryPool()
    {
        clean_pool();
    }

    MemoryPool::MemoryPool() = default;

    void MemoryPool::clean_pool()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (initialized_ || host_base_ || device_base_ || host_pool_ ||
            device_pool_)
        {
            // Restore the RMM default before destroying the resources it may
            // refer to; a default set by the application stays as it is.
            if (previous_rmm_default_)
            {
                set_rmm_default(
                    std::move(*previous_rmm_default_));
                previous_rmm_default_.reset();
            }
            host_stats_adaptor_.reset();
            host_pool_.reset();
            host_base_.reset();
            device_stats_adaptor_.reset();
            device_pool_.reset();
            device_base_.reset();
            pooled_device_allocations_.clear();
            pooled_host_allocations_.clear();
            use_pool_ = true;
            initialized_ = false;
        }
    }

} // namespace heongpu
