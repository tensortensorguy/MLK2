// MLK+ buffers and memory spaces (Rule 23: separate physical layer).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"

namespace mlk {

/// Memory spaces (realization spec §5.5; spec §8.8 memory.place).
enum class MemorySpace : uint8_t {
    Register = 0,
    Scratch,
    Shared,
    L1,
    L2,
    DRAM,
    HBM,
    Host,
    Persistent,
};

const char* memorySpaceName(MemorySpace s) noexcept;

/// Aligned, typed byte buffer (Rule 110: allocation failure handled —
/// bad_alloc is terminal in a no-exceptions build and treated as the
/// unrecoverable state defined by the runtime spec).
class Buffer {
public:
    Buffer() = default;
    Buffer(MemorySpace space, std::size_t bytes, std::size_t alignment =
               constants::kDefaultAlignmentBytes)
        : space_(space), bytes_(bytes) {
        data_.reset(new unsigned char[bytes + alignment]);
        const auto addr = reinterpret_cast<std::uintptr_t>(data_.get());
        const auto misalign = addr % alignment;
        ptr_ = data_.get() + (misalign == 0 ? 0 : alignment - misalign);
    }

    [[nodiscard]] void* data() noexcept { return ptr_; }
    [[nodiscard]] const void* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] MemorySpace space() const noexcept { return space_; }
    [[nodiscard]] float* asF32() noexcept {
        return static_cast<float*>(static_cast<void*>(ptr_));
    }
    [[nodiscard]] double* asF64() noexcept {
        return static_cast<double*>(static_cast<void*>(ptr_));
    }

private:
    MemorySpace space_{MemorySpace::DRAM};
    std::size_t bytes_{0};
    std::unique_ptr<unsigned char[]> data_;
    unsigned char* ptr_{nullptr};
};

}  // namespace mlk
