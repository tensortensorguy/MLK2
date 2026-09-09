// MLK+ generic hash functors for OpenHashMap/OpenHashSet (Rule 17).
#pragma once

#include "mlk/core/hash.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace mlk {

/// Identity-style hash for types directly hashable by mlk::hash* helpers.
struct FnvHashGeneric {
    [[nodiscard]] HashValue operator()(HashValue v) const noexcept {
        return v ^ (v >> 32);
    }
};

struct FnvHashU32 {
    [[nodiscard]] HashValue operator()(uint32_t v) const noexcept {
        return hashU64(v);
    }
};

struct FnvHashU64 {
    [[nodiscard]] HashValue operator()(uint64_t v) const noexcept {
        return hashU64(v);
    }
};

struct FnvHashPairU32 {
    [[nodiscard]] HashValue operator()(
        const std::pair<uint32_t, uint32_t>& p) const noexcept {
        return hashCombine(hashU64(p.first), hashU64(p.second));
    }
};

/// Primary template: falls back to text hash for string-like keys.
template <typename T, typename = void>
struct FnvHash {
    [[nodiscard]] HashValue operator()(const T& v) const noexcept {
        return hashText(v);
    }
};

template <typename T>
struct FnvHash<T, std::void_t<decltype(hashU64(uint64_t{}))>> {
    // Prefer a member/raw hash when T is an integral or enum type.
    [[nodiscard]] HashValue operator()(const T& v) const noexcept {
        if constexpr (std::is_enum_v<T>) {
            return hashU64(static_cast<uint64_t>(v));
        } else if constexpr (std::is_integral_v<T> || std::is_pointer_v<T>) {
            return hashU64(static_cast<uint64_t>(v));
        } else {
            return hashText(v);
        }
    }
};

}  // namespace mlk
