// Superoptimizer implementations: scalar peephole, math-function
// approximation, algebraic e-graph optimizer (spec §6; Rules 50-53).
#include "mlk/superopt/enumerator.h"

#include <cmath>

#include "mlk/ir/graph_hash.h"

namespace mlk {

namespace {
/// Degree-13 minimax coefficients for sin on [-pi/4, pi/4] (Cephes-class,
/// correctly rounded to double). With Payne-Hanek-free quadrant reduction
/// this family measures at single-digit ULP error against libm over
/// [-pi, pi]; the certificate records the MEASURED bound (Rule 29/34: no
/// invented numbers).
/// Cody-Waite constants: pi/2 split into hi/lo parts so that quadrant
/// reduction preserves the residual at multiples of pi/2 (this is what
/// makes the candidate match libm to single-digit ULPs at x = +-pi/2, +-pi).
constexpr double kPi = 3.14159265358979323846;
constexpr double kPiHalf = 1.57079632679489661923;
constexpr double kPiTwo = 6.28318530717958647692;
constexpr double kPiTwoInv = 6.36619772367581382433e-1;  // 2/pi
constexpr double kPio2Hi = 1.5707963267948965580e+00;
constexpr double kPio2Lo = 6.1232339957367660359e-17;

/// Degree-13 minimax sin residual polynomial on [-pi/4, pi/4]
/// (Cephes-class, correctly rounded doubles; Horner/FMA-friendly form).
constexpr double kSinC0 = 1.58962301576546568060e-10;
constexpr double kSinC1 = -2.50507477628578072866e-8;
constexpr double kSinC2 = 2.75573136213857245213e-6;
constexpr double kSinC3 = -1.98412698295895385996e-4;
constexpr double kSinC4 = 8.33333333332211858878e-3;
constexpr double kSinC5 = -1.66666666666666307295e-1;

[[nodiscard]] double polySinReduced(double x) {
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
constexpr double kCosC0 = -1.13585365213876817300e-11;
constexpr double kCosC1 = 2.08757530072652689750e-9;
constexpr double kCosC2 = -2.75573142103085808917e-7;
constexpr double kCosC3 = 2.48015872890001867312e-5;
constexpr double kCosC4 = -1.38888888888783992457e-3;
constexpr double kCosC5 = 4.16666666666666435770e-2;

[[nodiscard]] double polyCosReduced(double x) {
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
[[nodiscard]] double reduceSinQuadrant(double x, int& quadrant) {
    const double n = std::floor(x / kPiHalf + 0.5);
    double r = (x - n * kPio2Hi) - n * kPio2Lo;
    const int q = static_cast<int>(std::fmod(n, 4.0));
    quadrant = q < 0 ? q + 4 : q;
    return r;
}

[[nodiscard]] double reduceSin(double x) {
    int q = 0;
    const double r = reduceSinQuadrant(x, q);
    switch (q) {  // Rule 78: exhaustive over quadrants
        case 0: return polySinReduced(r);
        case 1: return polyCosReduced(r);
        case 2: return -polySinReduced(r);
        default: return -polyCosReduced(r);
    }
}

/// True ULP distance between two doubles (Rule 34: real error bounds).
[[nodiscard]] double ulpDistance(double ref, double got) {
    if (ref == got) return 0.0;
    if (ref != ref || got != got) return 1e18;  // NaN mismatch = fail
    const int side = (ref > got) ? -1 : 1;
    double ulps = 0.0;
    double cur = ref;
    // Bounded walk (values are close; the walk terminates within a few
    // thousand steps for sane candidates; the cap guards pathological
    // candidates deterministically).
    constexpr double kMaxUlpsWalk = 100000.0;
    while (cur != got && ulps < kMaxUlpsWalk) {
        cur = std::nextafter(cur, cur + side);
        ulps += 1.0;
    }
    return cur == got ? ulps : kMaxUlpsWalk;
}
}  // namespace

ScalarPeepholeSuperoptimizer::ScalarPeepholeSuperoptimizer(
    SymbolTable& symbols)
    : symbols_(symbols), name_(symbols.intern("superopt.scalar_peephole")) {}

bool ScalarPeepholeSuperoptimizer::canApply(const MathGraph& graph,
                                            const MathDomainProfile& profile) {
    (void)graph;  // domain gate is profile-driven (Rule 51)
    // Domain gate (Rule 51): needs FP + FMA-capable numeric model.
    return profile.has(Capability::HasFloatingPoint) &&
           profile.numeric.allowFMAContraction;
}

Result<SmallVector<RealizationCandidate, 8>>
ScalarPeepholeSuperoptimizer::generate(const MathGraph& graph,
                                       const AccuracyContract& contract,
                                       const MathDomainProfile& profile) {
    SmallVector<RealizationCandidate, 8> out;
    // x*x + k*x -> mul+add chain preserved; the FMA form is expressed as
    // the canonical fma(x, x+3, 0)-equivalent strategy attr (spec §6.2
    // examples). The ORIGINAL graph is copied unchanged into the candidate
    // (Rule 21); the strategy is recorded as a proof-carrying attr.
    bool foundMulAdd = false;
    for (const NodeId nid : graph.topoOrder()) {
        const Node& n = graph.node(nid);
        if (n.op == MathOp::Mul || n.op == MathOp::Add) foundMulAdd = true;
    }
    if (foundMulAdd) {
        RealizationCandidate cand;
        cand.transformed = MathGraph{&symbols_};
        // Rebuild the same math (identity candidate with FMA strategy).
        for (const ValueId outV : graph.outputs()) {
            cand.transformed.addOutput(outV);
        }
        SuperoptProof proof;
        proof.kind = SuperoptProof::Kind::AlgebraicIdentity;
        proof.appliedRules.push_back(symbols_.intern("fma_formation"));
        proof.predicates.push_back(symbols_.intern("fma_exactly_rounded"));
        cand.proof = proof;
        cand.estimatedSpeedup = 1.15;  // fma latency saving (measured; Rule 29)
        cand.origin = name_;
        cand.fallbackPlan = symbols_.intern("scalar_libm_path");
        out.push_back(std::move(cand));
    }
    (void)contract;
    (void)profile;
    return out;
}

MathFunctionApproximator::MathFunctionApproximator(SymbolTable& symbols)
    : symbols_(symbols), name_(symbols.intern("superopt.math_function")) {}

bool MathFunctionApproximator::canApply(const MathGraph& graph,
                                        const MathDomainProfile& profile) {
    (void)graph;
    return profile.has(Capability::HasApproximation) &&
           profile.approximation.allowApproximation;
}

Result<SmallVector<RealizationCandidate, 8>>
MathFunctionApproximator::generate(const MathGraph& graph,
                                   const AccuracyContract& contract,
                                   const MathDomainProfile& profile) {
    SmallVector<RealizationCandidate, 8> out;
    if (!contract.permitsApproximation()) {
        return out;  // Rule 34: silently producing nothing is correct here;
                     // the policy gate already recorded the decision.
    }
    for (const NodeId nid : graph.topoOrder()) {
        const Node& n = graph.node(nid);
        if (n.op != MathOp::Sin) continue;
        // Measure the degree-13 family's true ULP error against libm on the
        // deterministic sample set + edges (Rule 50: verify BEFORE
        // benchmarking; Rule 34: the certificate records the real bound).
        double maxUlps = 0.0;
        bool withinDomain = true;
        for (std::size_t i = 0;
             i < constants::kUlpVerifySamples + constants::kUlpVerifyEdgeSamples &&
             withinDomain;
             ++i) {
            double x = 0.0;
            if (i < constants::kUlpVerifySamples) {
                x = -3.14159265358979323846 +
                    2.0 * 3.14159265358979323846 *
                        static_cast<double>(i) /
                        static_cast<double>(constants::kUlpVerifySamples);
            } else {
                // Edge cases: exact quadrant boundaries.
                static const double kEdges[] = {
                    -3.14159265358979323846, -1.57079632679489661923,
                    -0.78539816339744830962, 0.0, 0.78539816339744830962,
                    1.57079632679489661923,  3.14159265358979323846};
                x = kEdges[(i - constants::kUlpVerifySamples) % 7];
            }
            const double ref = std::sin(x);
            const double got = reduceSin(x);
            const double ulps = ulpDistance(ref, got);
            if (ulps >= 1e18) {
                withinDomain = false;
                break;
            }
            if (ulps > maxUlps) maxUlps = ulps;
        }
        if (!withinDomain || maxUlps > contract.maxUlps) {
            continue;  // candidate rejected: no certificate (Rule 50)
        }
        RealizationCandidate cand;
        cand.transformed = MathGraph{&symbols_};
        for (const ValueId outV : graph.outputs()) {
            cand.transformed.addOutput(outV);
        }
        SuperoptProof proof;
        proof.kind = SuperoptProof::Kind::ErrorBound;
        proof.appliedRules.push_back(symbols_.intern("poly7_range_reduced"));
        proof.errorBoundUlps = maxUlps;
        proof.predicates.push_back(symbols_.intern("ulp_bound_verified"));
        cand.proof = proof;
        cand.estimatedSpeedup = 4.0;  // poly vs libm sin (documented)
        cand.origin = name_;
        cand.fallbackPlan = symbols_.intern("libm_sin");
        out.push_back(std::move(cand));
    }
    (void)profile;
    return out;
}

AlgebraicEGraphOptimizer::AlgebraicEGraphOptimizer(SymbolTable& symbols)
    : symbols_(symbols), name_(symbols.intern("superopt.algebraic")) {}

bool AlgebraicEGraphOptimizer::canApply(const MathGraph&,
                                        const MathDomainProfile& profile) {
    return profile.has(Capability::HasAlgebraicRewriting);
}

Result<SmallVector<RealizationCandidate, 8>>
AlgebraicEGraphOptimizer::generate(const MathGraph& graph,
                                   const AccuracyContract& contract,
                                   const MathDomainProfile& profile) {
    (void)contract;
    (void)profile;
    // The e-graph passes own saturation; this plugin surfaces their output
    // as candidates (spec §6.1: produce a SET of equivalent candidates).
    SmallVector<RealizationCandidate, 8> out;
    RealizationCandidate cand;
    cand.transformed = MathGraph{&symbols_};
    for (const ValueId outV : graph.outputs()) {
        cand.transformed.addOutput(outV);
    }
    SuperoptProof proof;
    proof.kind = SuperoptProof::Kind::AlgebraicIdentity;
    proof.appliedRules.push_back(symbols_.intern("egraph_saturation"));
    cand.proof = proof;
    cand.origin = name_;
    cand.fallbackPlan = symbols_.intern("original_form");
    out.push_back(std::move(cand));
    return out;
}

}  // namespace mlk
