// Tensor pass tests: transpose elimination, einsum lowering (Rule 33/§8.5).
#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/pass/register_all.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/type/domain_profile.h"

#include "mlk_test.h"

namespace {
struct TEnv {
    mlk::SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::MathDomainProfile profile;
    mlk::AccuracyContract contract;
    mlk::PassContext ctx;
    TEnv() {
        mlk::passes::registerAllPasses(symbols);
        profile.name = symbols.intern("test");
        profile.capabilities.set(mlk::Capability::HasTensorDomain);
        profile.capabilities.set(mlk::Capability::HasKernelFusion);
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier1;
    }
    mlk::Pass* pass(const char* n) {
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(n));
    }
};
}  // namespace

MLK_TEST(tensor, transpose_transpose_eliminated) {
    TEnv env;
    mlk::GraphBuilder b(env.symbols);
    auto a = b.placeholder("A",
        mlk::MathType::tensorValue(mlk::Dtype::F32, {2, 3}));
    auto t1 = b.op(mlk::MathOp::Transpose, {a});
    MLK_CHECK(t1.has_value());
    auto t2 = b.op(mlk::MathOp::Transpose, {*t1});
    MLK_CHECK(t2.has_value());
    b.output(*t2);
    MLK_CHECK(env.pass("type.infer")->run(env.ctx, b.graph()).has_value());
    auto r = env.pass("tensor.transpose_elim")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK_EQ(b.graph().representative(*t2), a);
}

MLK_TEST(tensor, einsum_ij_jk_ik_lowers_to_matmul) {
    TEnv env;
    mlk::GraphBuilder b(env.symbols);
    auto a = b.placeholder("A",
        mlk::MathType::tensorValue(mlk::Dtype::F32, {2, 3}));
    auto m = b.placeholder("B",
        mlk::MathType::tensorValue(mlk::Dtype::F32, {3, 4}));
    mlk::AttrList attrs;
    mlk::Attr eq;
    eq.name = env.symbols.intern("equation");
    eq.value = mlk::AttrValue{env.symbols.intern("ij,jk->ik")};
    attrs.push_back(eq);
    auto e = b.op(mlk::MathOp::Einsum, {a, m}, attrs);
    MLK_CHECK(e.has_value());
    b.output(*e);
    MLK_CHECK(env.pass("type.infer")->run(env.ctx, b.graph()).has_value());
    auto r = env.pass("tensor.einsum_lower")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    // The einsum node is dead; a MatMul replaced it (Rule 21 equivalence).
    const mlk::Value& v = b.graph().value(b.graph().representative(*e));
    MLK_CHECK(v.kind == mlk::ValueKind::NodeResult);
    MLK_CHECK(b.graph().node(v.producer).op == mlk::MathOp::MatMul);
}

MLK_TEST_MAIN("tensor")
