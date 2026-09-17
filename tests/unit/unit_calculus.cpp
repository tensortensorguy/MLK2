// Calculus tests: symbolic differentiation (spec §Pass: d/dx x² sin(x)).
#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/register_all.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/runtime/execution.h"
#include "mlk/type/domain_profile.h"

#include <cmath>

#include "mlk_test.h"

namespace {
struct CEnv {
    mlk::SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::MathDomainProfile profile;
    mlk::AccuracyContract contract;
    mlk::PassContext ctx;
    CEnv() {
        mlk::passes::registerAllPasses(symbols);
        profile.name = symbols.intern("test");
        profile.capabilities.set(mlk::Capability::HasNumericValues);
        profile.capabilities.set(mlk::Capability::HasFloatingPoint);
        profile.capabilities.set(mlk::Capability::HasDerivatives);
        profile.capabilities.set(mlk::Capability::HasCalculus);
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier2;
    }
};
}  // namespace

MLK_TEST(calculus, derivative_of_x_squared) {
    // d/dx x² = 2x
    CEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    const mlk::ValueId two = b.constant(2.0, f64);
    auto sq = b.op(mlk::MathOp::Pow, {x, two});
    MLK_CHECK(sq.has_value());
    auto d = b.op(mlk::MathOp::Derivative, {*sq, x});
    MLK_CHECK(d.has_value());
    b.output(*d);
    auto pass = mlk::PassRegistry::instance().byName(
        env.symbols, env.symbols.intern("calculus.derivative_symbolic"));
    MLK_CHECK(pass != nullptr);
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    // Evaluate d/dx x² at x=3 via the Tier 0 oracle: expect 2*3=6.
    mlk::SmallVector<double, 8> inputs{3.0};
    auto out = mlk::interpretGraph(b.graph(), inputs);
    MLK_CHECK(out.has_value());
    MLK_CHECK_NEAR(out->outputScalars[0], 6.0, 1e-9);
}

MLK_TEST(calculus, derivative_of_sin) {
    // d/dx sin(x) = cos(x); at x=0 expect 1.
    CEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    auto d = b.op(mlk::MathOp::Derivative, {*s, x});
    MLK_CHECK(d.has_value());
    b.output(*d);
    auto pass = mlk::PassRegistry::instance().byName(
        env.symbols, env.symbols.intern("calculus.derivative_symbolic"));
    MLK_CHECK(pass->run(env.ctx, b.graph()).has_value());
    mlk::SmallVector<double, 8> inputs{0.0};
    auto out = mlk::interpretGraph(b.graph(), inputs);
    MLK_CHECK(out.has_value());
    MLK_CHECK_NEAR(out->outputScalars[0], 1.0, 1e-12);
}

MLK_TEST(calculus, derivative_of_product_rule) {
    // d/dx x² sin(x) = 2x sin(x) + x² cos(x); at x=1: 2sin1 + cos1 ≈ 2.2216.
    CEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    const mlk::ValueId two = b.constant(2.0, f64);
    auto sq = b.op(mlk::MathOp::Pow, {x, two});
    MLK_CHECK(sq.has_value());
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    auto prod = b.op(mlk::MathOp::Mul, {*sq, *s});
    MLK_CHECK(prod.has_value());
    auto d = b.op(mlk::MathOp::Derivative, {*prod, x});
    MLK_CHECK(d.has_value());
    b.output(*d);
    auto pass = mlk::PassRegistry::instance().byName(
        env.symbols, env.symbols.intern("calculus.derivative_symbolic"));
    MLK_CHECK(pass->run(env.ctx, b.graph()).has_value());
    mlk::SmallVector<double, 8> inputs{1.0};
    auto out = mlk::interpretGraph(b.graph(), inputs);
    MLK_CHECK(out.has_value());
    MLK_CHECK_NEAR(out->outputScalars[0],
                   2.0 * std::sin(1.0) + std::cos(1.0), 1e-12);
}

MLK_TEST(calculus, capability_gate_blocks_derivatives) {
    // Part 0: absent capability => pass refuses (never assumes).
    CEnv env;
    env.profile.capabilities.clear(mlk::Capability::HasDerivatives);
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    auto d = b.op(mlk::MathOp::Derivative, {*s, x});
    MLK_CHECK(d.has_value());
    b.output(*d);
    auto pass = mlk::PassRegistry::instance().byName(
        env.symbols, env.symbols.intern("calculus.derivative_symbolic"));
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(!r.has_value());
    MLK_CHECK_EQ(r.error().code, mlk::ErrorCode::UnsupportedCapability);
}

MLK_TEST_MAIN("calculus")
