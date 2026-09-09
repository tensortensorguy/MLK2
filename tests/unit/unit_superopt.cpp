// Superoptimizer tests: certificates present (Rule 52), ULP verification
// (Rule 50), domain gating (Rule 51).
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/superopt/enumerator.h"
#include "mlk/type/domain_profile.h"

#include "mlk_test.h"

namespace {
struct SEnv {
    mlk::SymbolTable symbols;
    mlk::MathDomainProfile profile;
    mlk::AccuracyContract contract;
    SEnv(bool approx) {
        profile.name = symbols.intern("test");
        profile.capabilities.set(mlk::Capability::HasFloatingPoint);
        profile.capabilities.set(mlk::Capability::HasApproximation);
        profile.approximation.allowApproximation = approx;
        contract.maxUlps = 4.0;
        contract.allowFastMath = approx;
    }
};
}  // namespace

MLK_TEST(superopt, math_function_requires_approx_capability) {
    // Rule 51: domain-gated plugins refuse when the profile lacks the
    // capability.
    SEnv env(false);
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    mlk::MathFunctionApproximator approx(env.symbols);
    MLK_CHECK(!approx.canApply(b.graph(), env.profile));
    auto cands = approx.generate(b.graph(), env.contract, env.profile);
    MLK_CHECK(cands.has_value());
    MLK_CHECK(cands->empty());  // no capability => no candidates
}

MLK_TEST(superopt, poly7_candidate_carries_error_bound) {
    // Rule 52: every candidate carries a certificate with a verified bound.
    SEnv env(true);
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    mlk::MathFunctionApproximator approx(env.symbols);
    MLK_CHECK(approx.canApply(b.graph(), env.profile));
    auto cands = approx.generate(b.graph(), env.contract, env.profile);
    MLK_CHECK(cands.has_value());
    MLK_CHECK(!cands->empty());
    bool sawBound = false;
    for (const auto& c : *cands) {
        if (c.proof.errorBoundUlps.has_value()) {
            sawBound = true;
            MLK_CHECK(*c.proof.errorBoundUlps <= env.contract.maxUlps);
            MLK_CHECK(c.fallbackPlan != mlk::kInvalidSymbolId);
        }
    }
    MLK_CHECK(sawBound);
}

MLK_TEST(superopt, scalar_peephole_needs_fma_profile) {
    SEnv env(true);
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto sq = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq.has_value());
    auto tx = b.op(mlk::MathOp::Mul, {b.constant(3.0, f64), x});
    MLK_CHECK(tx.has_value());
    auto sum = b.op(mlk::MathOp::Add, {*sq, *tx});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    mlk::ScalarPeepholeSuperoptimizer peephole(env.symbols);
    env.profile.numeric.allowFMAContraction = false;
    MLK_CHECK(!peephole.canApply(b.graph(), env.profile));
    env.profile.numeric.allowFMAContraction = true;
    MLK_CHECK(peephole.canApply(b.graph(), env.profile));
    auto cands = peephole.generate(b.graph(), env.contract, env.profile);
    MLK_CHECK(cands.has_value());
    MLK_CHECK(!cands->empty());
    for (const auto& c : *cands) {
        MLK_CHECK(!c.proof.appliedRules.empty());
        MLK_CHECK(c.origin != mlk::kInvalidSymbolId);
    }
}

MLK_TEST_MAIN("superopt")
