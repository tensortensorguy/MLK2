// MLK+ monotonic arena (Rule 7: Zero-Allocation Hot Path).
//
// Both AOT and JIT compilers use std::pmr::monotonic_buffer_resource for IR
// allocation. Bulk-free after compilation. No malloc/free in the compiler hot
// path. Fallback materialization may allocate only through a controlled
// runtime path with explicit budgets.
#pragma once

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <new>
#include <utility>

namespace mlk {

/// RAII monotonic arena: allocations are bump-pointer fast, memory is
/// bulk-freed on destruction or reset(). Not thread-safe; each compilation
/// context owns its arena (Rule 12: thread-local allocation).
class MonotonicArena {
public:
    explicit MonotonicArena(std::size_t initialBytes = 1u << 16)
        : backingBytes_(initialBytes) {
        buffer_.reset(new std::byte[initialBytes]);
        resource_.reset(new std::pmr::monotonic_buffer_resource(
            buffer_.get(), initialBytes, std::pmr::null_memory_resource()));
    }

    MonotonicArena(const MonotonicArena&) = delete;
    MonotonicArena& operator=(const MonotonicArena&) = delete;

    [[nodiscard]] std::pmr::memory_resource* resource() noexcept {
        return resource_.get();
    }

    template <typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        std::pmr::polymorphic_allocator<T> alloc(resource_.get());
        T* ptr = alloc.allocate(1);
        ::new (static_cast<void*>(ptr)) T(std::forward<Args>(args)...);
        return ptr;
    }

    /// Release everything allocated so far. Only valid when no references
    /// remain (used between pipeline stages).
    void reset() { resource_->release(); }

    [[nodiscard]] std::size_t backingBufferBytes() const {
        return backingBytes_;
    }

private:
    std::unique_ptr<std::byte[]> buffer_;
    std::unique_ptr<std::pmr::monotonic_buffer_resource> resource_;
    std::size_t backingBytes_{0};
};

}  // namespace mlk
