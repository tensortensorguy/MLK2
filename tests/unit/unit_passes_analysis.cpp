// Analysis pass tests: type/shape/property/effect inference + verifier.
#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/pass/register_all.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/type/domain_profile.h"
#include "mlk/verifier/graph_verifier.h"

#include "mlk_test.h"

namespace {
struct Env {
    mlk::SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::MathDomainProfile profile;
    mlk::AccuracyContract contract;
    mlk::PassContext ctx;

    Env() {
        mlk::passes::registerAllPasses(symbols);
        profile.name = symbols.intern("test");
        profile.capabilities.set(mlk::Capability::HasNumericValues);
        profile.capabilities.set(mlk::Capability::HasFloatingPoint);
        profile.capabilities.set(mlk::Capability::HasTensorDomain);
        profile.lawCommutativeMul = mlk::TriState::True;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
    }
    mlk::Pass* pass(const char* name) {
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
    }
};
}  // namespace

MLK_TEST(analysis, type_infer_matmul_shape) {
    Env env;
    mlk::GraphBuilder b(env.symbols);
    // [m,k] x [k,n]: 2x3 * 3x4 -> 2x4
    auto a = b.placeholder("A", mlk::MathType::tensorValue(
        mlk::Dtype::F32, {2, 3}));
    auto m = b.placeholder("B", mlk::MathType::tensorValue(
        mlk::Dtype::F32, {3, 4}));
    auto mm = b.op(mlk::MathOp::MatMul, {a, m});
    MLK_CHECK(mm.has_value());
    b.output(*mm);
    auto r = env.pass("type.infer")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    const mlk::Value& out = b.graph().value(b.graph().outputs()[0]);
    MLK_CHECK(out.type.tensor.has_value());
    MLK_CHECK_EQ(out.type.tensor->shape.dim(0), 2);
    MLK_CHECK_EQ(out.type.tensor->shape.dim(1), 4);
}

MLK_TEST(analysis, type_infer_rejects_shape_mismatch_no_implicit_convert) {
    // Rule 39: 2x3 * 4x5 is an ERROR, not a coercion.
    Env env;
    mlk::GraphBuilder b(env.symbols);
    auto a = b.placeholder("A", mlk::MathType::tensorValue(
        mlk::Dtype::F32, {2, 3}));
    auto m = b.placeholder("B", mlk::MathType::tensorValue(
        mlk::Dtype::F32, {4, 5}));
    auto mm = b.op(mlk::MathOp::MatMul, {a, m});
    MLK_CHECK(mm.has_value());
    b.output(*mm);
    auto r = env.pass("type.infer")->run(env.ctx, b.graph());
    MLK_CHECK(!r.has_value());
    MLK_CHECK(!env.diag.entries().empty());
}

MLK_TEST(analysis, property_infer_matmul_non_commutative) {
    // Rule 33: matrix multiplication must not be assumed commutative.
    Env env;
    mlk::GraphBuilder b(env.symbols);
    auto a = b.placeholder("A", mlk::MathType::tensorValue(
        mlk::Dtype::F32, {2, 3}));
    auto m = b.placeholder("B", mlk::MathType::tensorValue(
        mlk::Dtype::F32, {3, 4}));
    auto mm = b.op(mlk::MathOp::MatMul, {a, m});
    MLK_CHECK(mm.has_value());
    b.output(*mm);
    MLK_CHECK(env.pass("type.infer")->run(env.ctx, b.graph()).has_value());
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    const mlk::Value& out = b.graph().value(b.graph().outputs()[0]);
    MLK_CHECK(out.facts.isFalse(mlk::PropertyId::Commutative));
}

MLK_TEST(analysis, property_infer_sin_range_and_period) {
    Env env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    MLK_CHECK(env.pass("type.infer")->run(env.ctx, b.graph()).has_value());
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    const mlk::Value& out = b.graph().value(b.graph().outputs()[0]);
    const mlk::Interval* range = out.facts.range();
    MLK_CHECK(range != nullptr);
    MLK_CHECK_NEAR(range->low, -1.0, 1e-12);
    MLK_CHECK_NEAR(range->high, 1.0, 1e-12);
    MLK_CHECK(out.facts.isTrue(mlk::PropertyId::Periodic));
}

MLK_TEST(analysis, verifier_catches_dangling_reference) {
    Env env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    auto x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    // Corrupt: fabricate a dangling node (simulated IR bug).
    b.graph().addOutput(9999);
    mlk::DiagnosticEngine diag;
    mlk::VerifyOptions opts;
    MLK_CHECK(!mlk::verifyGraph(b.graph(), &env.profile, opts, diag));
    MLK_CHECK(!diag.entries().empty());
}

MLK_TEST_MAIN("analysis")
