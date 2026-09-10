// MLK+ verified math implementation families (single source of truth).
//
// The superoptimizer (superopt/superopt.cpp) VERIFIES these families
// against libm and records the measured ULP bound in the candidate's
// accuracy certificate (Rule 34/50/52). The kernel executor
// (runtime/src/execution/kernel_buffers.cpp) EXECUTES them. One
// implementation, two consumers — no duplicated coefficient tables
// (Rule 77).
//
// Family "poly7" (Sin only today): degree-13 minimax residual polynomial
// on [-pi/4, pi/4] with Cody-Waite quadrant reduction. Measured at
// single-digit ULP error against libm over [-pi, pi] including quadrant
// edges (see tests/unit/unit_superopt.cpp for the certificate check).
#pragma once

#include <cmath>

namespace mlk::families {

/// Cody-Waite split constants: pi/2 = hi + lo with lo < 2^-33 so the
/// residual r = x - n*hi - n*lo keeps full precision at multiples of pi/2.
inline constexpr double kPiHalf = 1.57079632679489661923;
inline constexpr double kPio2Hi = 1.5707963267948965580e+00;
inline constexpr double kPio2Lo = 6.1232339957367660359e-17;

/// Degree-13 minimax sin residual polynomial on [-pi/4, pi/4]
/// (Cephes-class, correctly rounded doubles; Horner/FMA-friendly form).
inline constexpr double kSinC0 = 1.58962301576546568060e-10;
inline constexpr double kSinC1 = -2.50507477628578072866e-8;
inline constexpr double kSinC2 = 2.75573136213857245213e-6;
inline constexpr double kSinC3 = -1.98412698295895385996e-4;
inline constexpr double kSinC4 = 8.33333333332211858878e-3;
inline constexpr double kSinC5 = -1.66666666666666307295e-1;

[[nodiscard]] inline double polySinReduced(double x) noexcept {
    const double r2 = x * x;
    double p = kSinC0;
    p = p * r2 + kSinC1;
    p = p * r2 + kSinC2;
    p = p * r2 + kSinC3;
    p = p * r2 + kSinC4;
    p = p * r2 + kSinC5;
    return x + x * r2 * p;
}

/// Degree-12 minimax cos residual polynomial on [-pi/4, pi/4].
inline constexpr double kCosC0 = -1.13585365213876817300e-11;
inline constexpr double kCosC1 = 2.08757530072652689750e-9;
inline constexpr double kCosC2 = -2.75573142103085808917e-7;
inline constexpr double kCosC3 = 2.48015872890001867312e-5;
inline constexpr double kCosC4 = -1.38888888888783992457e-3;
inline constexpr double kCosC5 = 4.16666666666666435770e-2;

[[nodiscard]] inline double polyCosReduced(double x) noexcept {
    const double r2 = x * x;
    double p = kCosC0;
    p = p * r2 + kCosC1;
    p = p * r2 + kCosC2;
    p = p * r2 + kCosC3;
    p = p * r2 + kCosC4;
    p = p * r2 + kCosC5;
    return 1.0 - r2 * 0.5 + r2 * r2 * p;
}

/// Cody-Waite quadrant reduction: n = round(x / (pi/2)),
/// r = x - n*(pi/2_hi) - n*(pi/2_lo) in [-pi/4, pi/4].
[[nodiscard]] inline double reduceSinQuadrant(double x,
                                              int& quadrant) noexcept {
    const double n = std::floor(x / kPiHalf + 0.5);
    const double r = (x - n * kPio2Hi) - n * kPio2Lo;
    const int q = static_cast<int>(std::fmod(n, 4.0));
    quadrant = q < 0 ? q + 4 : q;
    return r;
}

/// Family "poly7": sin via quadrant-reduced minimax polynomials.
/// Verified accuracy contract: single-digit ULP vs libm on [-pi, pi]
/// (certificate = measured bound; Rule 34: no invented numbers).
[[nodiscard]] inline double polySin(double x) noexcept {
    int q = 0;
    const double r = reduceSinQuadrant(x, q);
    switch (q) {  // Rule 78: exhaustive over quadrants
        case 0: return polySinReduced(r);
        case 1: return polyCosReduced(r);
        case 2: return -polySinReduced(r);
        default: return -polyCosReduced(r);
    }
}

}  // namespace mlk::families
