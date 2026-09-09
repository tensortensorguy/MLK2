// Math pass unit tests (Rules 10, 22, 33, 87): idempotence, legality gates,
// canonicalization correctness.
#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/register_all.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/type/domain_profile.h"
#include "mlk/verifier/graph_verifier.h"

#include "mlk_test.h"

namespace {
struct PassEnv {
    mlk::SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::MathDomainProfile profile;
    mlk::AccuracyContract contract;
    mlk::PassContext ctx;

    PassEnv(bool allowReassoc = false) {
        mlk::passes::registerAllPasses(symbols);
        profile.name = symbols.intern("test");
        profile.capabilities.set(mlk::Capability::HasNumericValues);
        profile.capabilities.set(mlk::Capability::HasFloatingPoint);
        profile.capabilities.set(mlk::Capability::HasTensorDomain);
        profile.capabilities.set(mlk::Capability::HasKernelFusion);
        profile.capabilities.set(mlk::Capability::HasAlgebraicRewriting);
        profile.capabilities.set(mlk::Capability::HasDerivatives);
        profile.capabilities.set(mlk::Capability::HasApproximation);
        profile.lawCommutativeMul = mlk::TriState::True;
        profile.lawAssociativeAdd =
            allowReassoc ? mlk::TriState::True : mlk::TriState::Unknown;
        profile.numeric.allowReassociation = allowReassoc;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier1;
    }

    mlk::Pass* pass(const char* name) {
        return mlk::PassRegistry::instance().byName(symbols.intern(name));
    }
};
}  // namespace

MLK_TEST(math, constant_fold_arithmetic) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    auto two = b.constant(2.0, f64);
    auto three = b.constant(3.0, f64);
    auto sum = b.op(mlk::MathOp::Add, {two, three});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    auto pass = env.pass("math.constant_fold");
    MLK_CHECK(pass != nullptr);
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    // 2+3 folded to constant 5; the original add node is dead (Rule 21).
    const mlk::Value& v = b.graph().value(b.graph().representative(*sum));
    MLK_CHECK(v.kind == mlk::ValueKind::Constant);
    MLK_CHECK_NEAR(v.constant.f64, 5.0, 1e-12);
}

MLK_TEST(math, constant_fold_never_creates_nan) {
    // Rule 87: log(-1) is a domain violation; no fold, no NaN creation.
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    auto neg = b.constant(-1.0, f64);
    auto l = b.op(mlk::MathOp::Log, {neg});
    MLK_CHECK(l.has_value());
    b.output(*l);
    auto pass = env.pass("math.constant_fold");
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(!r->changed);
    const mlk::Value& v = b.graph().value(b.graph().representative(*l));
    MLK_CHECK(v.kind == mlk::ValueKind::NodeResult);  // unfolded
}

MLK_TEST(math, identity_elim_x_plus_zero) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto zero = b.constant(0.0, f64);
    auto sum = b.op(mlk::MathOp::Add, {x, zero});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    // property.infer first (identity elim requires facts)
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto pass = env.pass("math.identity_elim");
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK_EQ(b.graph().representative(*sum), x);
}

MLK_TEST(math, identity_elim_preserves_neg_zero) {
    // Rule 90: x + (-0.0) != x when x = -0.0 under sign preservation.
    PassEnv env;  // profile preserves negative zero
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto nz = b.constant(-0.0, f64);
    auto sum = b.op(mlk::MathOp::Add, {x, nz});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto r = env.pass("math.identity_elim")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(!r->changed);  // NOT eliminated
    MLK_CHECK_EQ(b.graph().representative(*sum), *sum);
}

MLK_TEST(math, cse_merges_duplicate_subtrees) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto sq1 = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq1.has_value());
    auto sq2 = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq2.has_value());
    auto sum = b.op(mlk::MathOp::Add, {*sq1, *sq2});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    auto r = env.pass("math.cse")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK_EQ(b.graph().representative(*sq2),
                 b.graph().representative(*sq1));
}

MLK_TEST(math, dce_removes_dead_pure_nodes_only) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto dead = b.op(mlk::MathOp::Sin, {x});  // never used
    MLK_CHECK(dead.has_value());
    auto used = b.op(mlk::MathOp::Cos, {x});
    MLK_CHECK(used.has_value());
    b.output(*used);
    auto r = env.pass("math.dce")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK(b.graph().isNodeDead(b.graph().value(*dead).producer));
    MLK_CHECK(!b.graph().isNodeDead(b.graph().value(*used).producer));
}

MLK_TEST(math, passes_are_idempotent) {
    // Rule 10: running twice produces identical IR.
    PassEnv env;
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
    auto s = b.op(mlk::MathOp::Sin, {*sum});
    MLK_CHECK(s.has_value());
    b.output(*s);
    auto canon = env.pass("math.canonicalize");
    MLK_CHECK(canon != nullptr);
    MLK_CHECK(canon->run(env.ctx, b.graph()).has_value());
    const mlk::HashValue h1 = mlk::graphHash(b.graph());
    const uint32_t live1 = b.graph().liveNodeCount();
    MLK_CHECK(canon->run(env.ctx, b.graph()).has_value());
    const mlk::HashValue h2 = mlk::graphHash(b.graph());
    MLK_CHECK_EQ(h1, h2);
    MLK_CHECK_EQ(live1, b.graph().liveNodeCount());
}

MLK_TEST_MAIN("math")
