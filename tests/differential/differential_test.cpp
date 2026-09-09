// Rule 43: Tier 0 <-> Tier 1 <-> Tier 2 <-> Tier 3 comparisons on every
// supported profile. Divergence blocks merge.
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/pass/register_all.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/telemetry.h"
#include "mlk/type/domain_profile.h"

#include "mlk_test.h"

namespace {
mlk::MathGraph buildSinGraph(mlk::SymbolTable& symbols) {
    mlk::GraphBuilder b(symbols);
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
    return std::move(b.graph());
}
}  // namespace

MLK_TEST(differential, all_tiers_agree_on_sin_graph) {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::TelemetrySink telemetry;
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("scalar_f64");
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    mlk::AccuracyContract contract;
    mlk::ExecutionEngine engine(symbols, telemetry);
    for (const mlk::Tier tier : {mlk::Tier::Tier0, mlk::Tier::Tier1,
                                 mlk::Tier::Tier2, mlk::Tier::Tier3}) {
        mlk::MathGraph graph = buildSinGraph(symbols);
        mlk::SmallVector<double, 8> inputs{2.0};
        auto r = engine.execute(graph, profile, contract, tier, inputs);
        MLK_CHECK(r.has_value());
        if (r.has_value()) {
            MLK_CHECK_NEAR(r->outputScalars[0], -0.5440211108893698, 1e-13);
        }
    }
}

MLK_TEST(differential, oracle_is_authoritative) {
    // Rule 85: Tier 0 reference semantics are the ground truth.
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto e = b.op(mlk::MathOp::Exp, {x});
    MLK_CHECK(e.has_value());
    b.output(*e);
    mlk::SmallVector<double, 8> inputs{1.0};
    auto out = mlk::interpretGraph(b.graph(), inputs);
    MLK_CHECK(out.has_value());
    MLK_CHECK_NEAR(out->outputScalars[0], 2.718281828459045, 1e-14);
}

MLK_TEST_MAIN("differential")
