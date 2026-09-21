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

MLK_TEST(tensor, contraction_path_matrix_chain_dp) {
    // Chain M1(10x5) * M2(5x50) * M3(50x1), built left-deep:
    //   mm1 = M1*M2, mm2 = mm1*M3.
    // Left-deep evaluation cost: 10*5*50 + 10*50*1 = 3000.
    // Optimal (right) evaluation: 5*50*1 + 10*5*1 = 300 — the DP must find
    // split k=2 at the root and record the per-subtree costs as attrs.
    // (The order itself is strategy data: realizing it reassociates FP
    // matmuls and needs a Rule 33 contract.)
    TEnv env;
    mlk::GraphBuilder b(env.symbols);
    auto m1 = b.placeholder("M1",
        mlk::MathType::tensorValue(mlk::Dtype::F64, {10, 5}));
    auto m2 = b.placeholder("M2",
        mlk::MathType::tensorValue(mlk::Dtype::F64, {5, 50}));
    auto m3 = b.placeholder("M3",
        mlk::MathType::tensorValue(mlk::Dtype::F64, {50, 1}));
    auto mm1 = b.op(mlk::MathOp::MatMul, {m1, m2});
    MLK_CHECK(mm1.has_value());
    auto mm2 = b.op(mlk::MathOp::MatMul, {*mm1, m3});
    MLK_CHECK(mm2.has_value());
    b.output(*mm2);
    MLK_CHECK(env.pass("type.infer")->run(env.ctx, b.graph()).has_value());
    auto pass = env.pass("tensor.contraction_path");
    MLK_CHECK(pass != nullptr);
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());

    const mlk::SymbolId splitName = env.symbols.intern("contraction_split");
    const mlk::SymbolId flopsName =
        env.symbols.intern("contraction_flops");
    // Root (mm2): covers matrices 1..3; the optimal top split is k=1
    // (left block {M1}, right block {M2,M3} = M1*(M2*M3)); cost 300.
    const mlk::Value& rootV = b.graph().value(b.graph().representative(*mm2));
    MLK_CHECK(rootV.kind == mlk::ValueKind::NodeResult);
    const mlk::Node& rootN = b.graph().node(rootV.producer);
    const mlk::AttrValue* rootSplit = mlk::findAttr(rootN.attrs, splitName);
    MLK_CHECK(rootSplit != nullptr);
    MLK_CHECK(std::holds_alternative<int64_t>(rootSplit->v));
    MLK_CHECK_EQ(std::get<int64_t>(rootSplit->v), 1);
    const mlk::AttrValue* rootFlops = mlk::findAttr(rootN.attrs, flopsName);
    MLK_CHECK(rootFlops != nullptr);
    MLK_CHECK(std::holds_alternative<double>(rootFlops->v));
    MLK_CHECK(std::get<double>(rootFlops->v) == 300.0);
    // Interior node (mm1): covers matrices 1..2, split 1, cost 2500.
    const mlk::Value& innerV =
        b.graph().value(b.graph().representative(*mm1));
    const mlk::Node& innerN = b.graph().node(innerV.producer);
    const mlk::AttrValue* innerFlops = mlk::findAttr(innerN.attrs, flopsName);
    MLK_CHECK(innerFlops != nullptr);
    MLK_CHECK(std::get<double>(innerFlops->v) == 2500.0);
    const mlk::AttrValue* innerSplit = mlk::findAttr(innerN.attrs, splitName);
    MLK_CHECK(innerSplit != nullptr);
    MLK_CHECK_EQ(std::get<int64_t>(innerSplit->v), 1);

    // Idempotent: a second run adds nothing new (attrs are set once).
    auto r2 = pass->run(env.ctx, b.graph());
    MLK_CHECK(r2.has_value());
    const mlk::AttrValue* rootSplit2 = mlk::findAttr(rootN.attrs, splitName);
    MLK_CHECK(rootSplit2 == rootSplit);
}

MLK_TEST_MAIN("tensor")
