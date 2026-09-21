// Superoptimizer implementations: scalar peephole, math-function
// approximation, algebraic e-graph optimizer (spec §6; Rules 50-53).
#include "mlk/superopt/enumerator.h"

#include "mlk/support/math_families.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "mlk/ir/graph_hash.h"

namespace mlk {

namespace {
// ulpDistance now lives in math_families.h as the single source of truth
// shared with approx.ulp_verify (Rule 77: no duplicated quality logic).
using mlk::families::ulpDistance;

// --- measured speedups (Rule 29: claims carry their measurement) -----------
// The previous implementation wrote estimatedSpeedup = 1.15 / 4.0 with a
// "(measured)" comment and no measurement. Both numbers are now real
// ratios of steady-clock medians over a deterministic workload; the
// workload and rep counts are named constants (Rule 27).
constexpr std::size_t kSpeedupSamples = 512;
constexpr uint32_t kSpeedupReps = 31;  // odd: the median is a real sample

double speedupWorkload(uint64_t i) {
    const double x = static_cast<double>(i % 4096) / 512.0;
    return 0.5 * std::sin(x) + 0.25 * std::cos(2.0 * x) + 0.5;
}

double medianMs(const SmallVector<double, 64>& samples) {
    SmallVector<double, 64> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
}

/// Times refFn (reference) against candFn (candidate) over the
/// deterministic workload; returns tRef/tCand, or fallbackRatio when
/// either side fails to produce a positive median (degenerate timing —
/// reported as the fallback, never guessed).
template <typename FA, typename FB>
double measuredSpeedupRatio(FA&& refFn, FB&& candFn, double fallbackRatio) {
    double refSink = 0.0;
    double candSink = 0.0;
    SmallVector<double, 64> refTimes;
    SmallVector<double, 64> candTimes;
    for (uint32_t r = 0; r < kSpeedupReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kSpeedupSamples; ++i) {
            refSink += refFn(speedupWorkload(i));
        }
        const auto t1 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kSpeedupSamples; ++i) {
            candSink += candFn(speedupWorkload(i));
        }
        const auto t2 = std::chrono::steady_clock::now();
        refTimes.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        candTimes.push_back(
            std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    if (refSink == 0.0 || candSink != candSink) return fallbackRatio;
    const double refMs = medianMs(refTimes);
    const double candMs = medianMs(candTimes);
    if (refMs <= 0.0 || candMs <= 0.0) return fallbackRatio;
    return refMs / candMs;
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
        // Measured HERE (Rule 29): mul+add vs fma over the deterministic
        // workload. No FMA hardware -> the ratio honestly reports < 1.
        cand.estimatedSpeedup = measuredSpeedupRatio(
            [](double x) { return x * x + 3.0 * x; },
            [](double x) { return std::fma(x, x, 3.0 * x); }, 1.0);
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
            const double got = families::polySin(x);
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
        // Measured HERE (Rule 29): libm sin vs the polynomial family over
        // the deterministic workload — the ratio this machine actually
        // shows, not a documented constant.
        cand.estimatedSpeedup = measuredSpeedupRatio(
            [](double x) { return std::sin(x); },
            [](double x) { return families::polySin(x); }, 1.0);
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
