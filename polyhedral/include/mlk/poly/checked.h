// MLK+ polyhedral engine — checked int64 arithmetic primitives.
//
// -Wpedantic forbids __int128 in ISO C++ (the build treats it as an error),
// so the engine uses GCC overflow-checking builtins on int64 with a bounded
// input magnitude (kRationalMagnitudeLimit = 2^30): every documented product
// of two in-range values fits int64 (2^60 < 2^63), and every accumulation
// is overflow-checked (Rule 6: failures are Results, Rule 73: no silent
// wraparound).
#pragma once

#include <cstdint>

#include "mlk/core/result.h"
#include "mlk/poly/rational.h"

namespace mlk::poly::checked {

/// a + b into out; false on int64 overflow.
[[nodiscard]] inline bool add(int64_t a, int64_t b, int64_t* out) noexcept {
    return !__builtin_add_overflow(a, b, out);
}

/// a - b into out; false on int64 overflow.
[[nodiscard]] inline bool sub(int64_t a, int64_t b, int64_t* out) noexcept {
    return !__builtin_sub_overflow(a, b, out);
}

/// a * b into out; false on int64 overflow.
[[nodiscard]] inline bool mul(int64_t a, int64_t b, int64_t* out) noexcept {
    return !__builtin_mul_overflow(a, b, out);
}

/// Checked + magnitude-limited add: fails when the result exceeds the
/// engine magnitude limit (keeps every later product inside int64).
[[nodiscard]] inline Result<int64_t> addLimited(int64_t a,
                                                int64_t b) noexcept {
    int64_t r = 0;
    if (!add(a, b, &r) || r > kRationalMagnitudeLimit ||
        r < -kRationalMagnitudeLimit) {
        return err(ErrorCode::ResourceExhausted,
                   "checked add exceeds engine magnitude limit");
    }
    return r;
}

/// Checked + magnitude-limited sub.
[[nodiscard]] inline Result<int64_t> subLimited(int64_t a,
                                                int64_t b) noexcept {
    int64_t r = 0;
    if (!sub(a, b, &r) || r > kRationalMagnitudeLimit ||
        r < -kRationalMagnitudeLimit) {
        return err(ErrorCode::ResourceExhausted,
                   "checked sub exceeds engine magnitude limit");
    }
    return r;
}

/// Checked + magnitude-limited mul.
[[nodiscard]] inline Result<int64_t> mulLimited(int64_t a,
                                                int64_t b) noexcept {
    int64_t r = 0;
    if (!mul(a, b, &r) || r > kRationalMagnitudeLimit ||
        r < -kRationalMagnitudeLimit) {
        return err(ErrorCode::ResourceExhausted,
                   "checked mul exceeds engine magnitude limit");
    }
    return r;
}

}  // namespace mlk::poly::checked
