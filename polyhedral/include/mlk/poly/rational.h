// MLK+ polyhedral engine — exact rational arithmetic kernel.
//
// The polyhedral engine needs exact arithmetic (Rule 90: numeric semantics
// preserved exactly; Rule 33: no legality decision on rounded values).
// Rational values are int64/int64, kept normalized (gcd(|num|, den) == 1,
// den > 0). Intermediates are computed in __int128 and range-checked, so
// every operation is total and reports overflow through Result<T> instead
// of throwing (Rule 6) or wrapping (Rule 73: no silent bugs).
//
// Cold-path component: used by scheduling/dependence/codegen at compile
// time, never inside kernel execution hot loops (Rule 81 does not apply to
// compile-time arithmetic; correctness does).
#pragma once

#include <cstdint>

#include "mlk/core/result.h"

namespace mlk::poly {

/// Magnitude ceiling for engine inputs. Loop bounds, coefficients, and
/// schedule constants beyond this bound are rejected at SCoP extraction
/// (Rule 10: explicit budgets, not evaluation stalls). The bound is 2^30
/// so every documented product of two in-range values stays inside int64
/// (see checked.h; -Wpedantic forbids __int128).
inline constexpr int64_t kRationalMagnitudeLimit = int64_t{1} << 30;

/// Normalized exact rational: num/den with den > 0 and gcd == 1.
struct Rational {
    int64_t num{0};
    int64_t den{1};

    [[nodiscard]] static Rational fromInt(int64_t v) noexcept {
        return Rational{v, 1};
    }
    /// Builds and normalizes num/den; fails on zero denominator or
    /// out-of-range parts (Rule 6: no silent wraparound).
    [[nodiscard]] static Result<Rational> fromParts(int64_t n,
                                                    int64_t d) noexcept;
    [[nodiscard]] bool isZero() const noexcept { return num == 0; }
    [[nodiscard]] bool isInt() const noexcept { return den == 1; }
    [[nodiscard]] int32_t sign() const noexcept {
        return num > 0 ? 1 : (num < 0 ? -1 : 0);
    }
};

/// Total ordering (no overflow: cross-multiplied in __int128).
[[nodiscard]] bool ratLess(const Rational& a, const Rational& b) noexcept;
[[nodiscard]] bool ratEqual(const Rational& a, const Rational& b) noexcept;

/// Exact arithmetic; fails with ErrorCode::ResourceExhausted on intermediate
/// overflow past kRationalMagnitudeLimit (Rule 67: actionable diagnostics).
[[nodiscard]] Result<Rational> ratAdd(const Rational& a,
                                      const Rational& b) noexcept;
[[nodiscard]] Result<Rational> ratSub(const Rational& a,
                                      const Rational& b) noexcept;
[[nodiscard]] Result<Rational> ratMul(const Rational& a,
                                      const Rational& b) noexcept;
[[nodiscard]] Result<Rational> ratDiv(const Rational& a,
                                      const Rational& b) noexcept;
[[nodiscard]] Result<Rational> ratNeg(const Rational& a) noexcept;

/// Floor/ceil toward integers; fails when the value has no int64 ceiling.
[[nodiscard]] Result<int64_t> ratFloor(const Rational& r) noexcept;
[[nodiscard]] Result<int64_t> ratCeil(const Rational& r) noexcept;

/// gcd on non-negative values; gcd(0, 0) == 0.
[[nodiscard]] int64_t ratGcd(int64_t a, int64_t b) noexcept;

}  // namespace mlk::poly
