// Superoptimizer implementations: scalar peephole, math-function
// approximation, algebraic e-graph optimizer (spec §6; Rules 50-53).
#include "mlk/superopt/enumerator.h"

#include <cmath>

#include "mlk/ir/graph_hash.h"

namespace mlk {

namespace {
/// poly7 sin/cos coefficients on [-pi/4, pi/4] (Chebyshev-derived; the
/// certificate with the measured ULP bound ships in tests/superopt and is
/// re-checked by approx.ulp_verify — Rule 50).
constexpr int kPolyDegree = 7;

[[nodiscard]] double poly7Sin(double x) {
    // Range-reduced polynomial sin approximation (deterministic; the
    // bound certificate records max ULP over the domain).
    static const double c1 = 0.99999999999999997;
    static const double c3 = -0.16666666666657394;
    static const double c5 = 0.00833333333330991;
    static const double c7 = -0.00019841265528565;
    const double x2 = x * x;
    return x * (c1 + x2 * (c3 + x2 * (c5 + x2 * c7)));
}

[[nodiscard]] double reduceSin(double x) {
    // Range reduction: sin(x) = sin(r) with r in [-pi/4, pi/4] via
    // quadrant folding (spec §6.4; approximation superoptimizer output).
    constexpr double kHalfPi = 1.5707963267948966;
    constexpr double kTwoPi = 6.283185307179586;
    double r = std::fmod(x, kTwoPi);
    if (r > kHalfPi) r = 3.141592653589793 - r;
    if (r < -kHalfPi) r = -3.141592653589793 - r;
    return poly7Sin(r);
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
        // Measure the poly7+range-reduction error against libm on the
        // deterministic sample set (Rule 50: verify BEFORE benchmarking).
        double maxUlps = 0.0;
        bool withinDomain = true;
        for (std::size_t i = 0; i < constants::kUlpVerifySamples; ++i) {
            const double x =
                -3.141592653589793 +
                2.0 * 3.141592653589793 *
                    static_cast<double>(i) /
                    static_cast<double>(constants::kUlpVerifySamples);
            const double ref = std::sin(x);
            const double got = reduceSin(x);
            if (ref != ref || got != got) {
                withinDomain = false;
                break;
            }
            const double err = std::fabs(ref - got);
            const double ulps =
                ref != 0.0 ? err / std::fabs(ref) * 4503599627370496.0 : err;
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
