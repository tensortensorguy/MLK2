// Exact rational arithmetic implementation (see rational.h).
//
// All intermediates are int64 with overflow-checked operations (checked.h).
// kRationalMagnitudeLimit = 2^30 guarantees cross-products fit int64.
#include "mlk/poly/rational.h"

#include "mlk/poly/checked.h"

namespace mlk::poly {

namespace {

/// Normalizes num/den: den > 0, gcd == 1, result magnitude-limited.
/// Preconditions: |num|, |den| fit int64 and den != 0 (callers guarantee
/// via checked products); results outside kRationalMagnitudeLimit fail so
/// every LATER cross-product stays inside int64 (Rule 10: bounded).
[[nodiscard]] Result<Rational> normalizeLimited(int64_t num,
                                                int64_t den) noexcept {
    if (den < 0) {
        num = -num;
        den = -den;
    }
    const int64_t an = num < 0 ? -num : num;
    const int64_t g = ratGcd(an, den);
    if (g > 1) {
        num /= g;
        den /= g;
    }
    if (num > kRationalMagnitudeLimit || num < -kRationalMagnitudeLimit ||
        den > kRationalMagnitudeLimit) {
        return err(ErrorCode::ResourceExhausted,
                   "rational result exceeds engine magnitude limit");
    }
    return Rational{num, den};
}

}  // namespace

int64_t ratGcd(int64_t a, int64_t b) noexcept {
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a < 0 ? -a : a;
}

Result<Rational> Rational::fromParts(int64_t n, int64_t d) noexcept {
    if (d == 0) {
        return err(ErrorCode::InvalidArgument,
                   "rational denominator is zero");
    }
    if (n > kRationalMagnitudeLimit || n < -kRationalMagnitudeLimit ||
        d > kRationalMagnitudeLimit || d < -kRationalMagnitudeLimit) {
        return err(ErrorCode::ResourceExhausted,
                   "rational parts exceed engine magnitude limit");
    }
    return normalizeLimited(n, d);
}

bool ratLess(const Rational& a, const Rational& b) noexcept {
    // a.num * b.den < b.num * a.den; inputs are magnitude-limited to 2^30
    // so each product fits int64 (2^60). The checked multiplies defend the
    // invariant; on impossible overflow the comparison degrades to false
    // deterministically (documented; unreachable for in-range inputs).
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!checked::mul(a.num, b.den, &lhs) ||
        !checked::mul(b.num, a.den, &rhs)) {
        return false;
    }
    return lhs < rhs;
}

bool ratEqual(const Rational& a, const Rational& b) noexcept {
    return a.num == b.num && a.den == b.den;
}

Result<Rational> ratAdd(const Rational& a, const Rational& b) noexcept {
    int64_t t1 = 0;
    int64_t t2 = 0;
    if (!checked::mul(a.num, b.den, &t1) ||
        !checked::mul(b.num, a.den, &t2)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational add cross-product overflow");
    }
    int64_t num = 0;
    if (!checked::add(t1, t2, &num)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational add numerator overflow");
    }
    int64_t den = 0;
    if (!checked::mul(a.den, b.den, &den)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational add denominator overflow");
    }
    return normalizeLimited(num, den);
}

Result<Rational> ratSub(const Rational& a, const Rational& b) noexcept {
    int64_t t1 = 0;
    int64_t t2 = 0;
    if (!checked::mul(a.num, b.den, &t1) ||
        !checked::mul(b.num, a.den, &t2)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational sub cross-product overflow");
    }
    int64_t num = 0;
    if (!checked::sub(t1, t2, &num)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational sub numerator overflow");
    }
    int64_t den = 0;
    if (!checked::mul(a.den, b.den, &den)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational sub denominator overflow");
    }
    return normalizeLimited(num, den);
}

Result<Rational> ratMul(const Rational& a, const Rational& b) noexcept {
    int64_t num = 0;
    int64_t den = 0;
    if (!checked::mul(a.num, b.num, &num) ||
        !checked::mul(a.den, b.den, &den)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational mul overflows int64");
    }
    return normalizeLimited(num, den);
}

Result<Rational> ratDiv(const Rational& a, const Rational& b) noexcept {
    if (b.num == 0) {
        return err(ErrorCode::InvalidArgument, "rational division by zero");
    }
    int64_t num = 0;
    int64_t den = 0;
    if (!checked::mul(a.num, b.den, &num) ||
        !checked::mul(a.den, b.num, &den)) {
        return err(ErrorCode::ResourceExhausted,
                   "rational div overflows int64");
    }
    return normalizeLimited(num, den);
}

Result<Rational> ratNeg(const Rational& a) noexcept {
    return Rational{-a.num, a.den};
}

Result<int64_t> ratFloor(const Rational& r) noexcept {
    // den > 0: floor(num/den) adjusts down when the remainder is nonzero
    // and the numerator is negative (C++ '/' truncates toward zero).
    int64_t q = r.num / r.den;
    if (r.num % r.den != 0 && r.num < 0) --q;
    return q;
}

Result<int64_t> ratCeil(const Rational& r) noexcept {
    const int64_t rem = r.num % r.den;
    int64_t q = r.num / r.den;
    if (rem != 0 && r.num > 0) ++q;
    return q;
}

}  // namespace mlk::poly
