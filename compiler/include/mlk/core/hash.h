// MLK+ stable hashing (Rule 24: stable content hashes; Rule 98: no
// assumptions about hash seeds or addresses). FNV-1a 64-bit is deterministic
// across runs, platforms, and processes, which is what serialization and
// cache keys require. std::hash is NOT stable and is forbidden for
// persisted artifacts.
#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

namespace mlk {

using HashValue = uint64_t;

inline constexpr HashValue kHashSeed = 0xcbf29ce484222325ULL;
inline constexpr HashValue kHashPrime = 0x100000001b3ULL;

[[nodiscard]] inline HashValue hashBytes(const void* data,
                                         std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    HashValue h = kHashSeed;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= kHashPrime;
    }
    return h;
}

[[nodiscard]] inline HashValue hashCombine(HashValue a, HashValue b) noexcept {
    // Order-sensitive combination; multiplying by the prime keeps entropy.
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a * kHashPrime;
}

[[nodiscard]] inline HashValue hashU64(uint64_t v) noexcept {
    return hashBytes(&v, sizeof(v));
}

[[nodiscard]] inline HashValue hashI64(int64_t v) noexcept {
    return hashBytes(&v, sizeof(v));
}

[[nodiscard]] inline HashValue hashF64(double v) noexcept {
    // Canonicalize NaN payloads so hashing is deterministic (Rule 98).
    uint64_t bits = 0;
    if (v != v) {
        bits = 0x7ff8000000000000ULL;  // canonical quiet NaN
    } else {
        std::memcpy(&bits, &v, sizeof(bits));
        // Normalize -0.0 to +0.0 so both hash identically.
        if (bits == 0x8000000000000000ULL) bits = 0;
    }
    return hashBytes(&bits, sizeof(bits));
}

[[nodiscard]] inline HashValue hashText(std::string_view s) noexcept {
    return hashBytes(s.data(), s.size());
}

}  // namespace mlk
