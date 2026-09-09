// E-graph tests (Rule 21): equivalence classes, identity rules, extraction.
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/pass/register_all.h"

#include "../src/passes/egraph/egraph.h"
#include "mlk_test.h"

MLK_TEST(egraph, identity_rules_union_classes) {
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    const mlk::ValueId one = b.constant(1.0, f64);
    auto m = b.op(mlk::MathOp::Mul, {x, one});  // x*1 -> x
    MLK_CHECK(m.has_value());
    b.output(*m);

    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("test");
    profile.numeric.ieee754 = true;
    mlk::EGraph eg(profile);
    MLK_CHECK(eg.importGraph(b.graph(), b.graph().outputs()[0]).has_value());
    auto grew = eg.saturateOnce(symbols);
    MLK_CHECK(grew.has_value());
    MLK_CHECK(*grew);
    // Extract: cheapest form in the merged class is the leaf x.
    mlk::MathGraph out{&symbols};
    auto extracted = eg.extract(symbols, out, eg.find(0));
    MLK_CHECK(extracted.has_value());
}

MLK_TEST(egraph, pow_square_rule) {
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    const mlk::ValueId two = b.constant(2.0, f64);
    auto p = b.op(mlk::MathOp::Pow, {x, two});  // x^2 <-> x*x
    MLK_CHECK(p.has_value());
    b.output(*p);

    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("test");
    profile.numeric.ieee754 = true;
    mlk::EGraph eg(profile);
    MLK_CHECK(eg.importGraph(b.graph(), b.graph().outputs()[0]).has_value());
    MLK_CHECK(eg.saturateOnce(symbols).has_value());
    MLK_CHECK(eg.saturateOnce(symbols).has_value());
    mlk::MathGraph out{&symbols};
    auto extracted = eg.extract(symbols, out, 0);
    MLK_CHECK(extracted.has_value());
    // Extraction must terminate and produce a connected graph.
    MLK_CHECK(out.numValues() >= 1);
}

MLK_TEST(egraph, saturation_is_budgeted) {
    // Rule 10: strict budget — no runaway growth.
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto e1 = b.op(mlk::MathOp::Exp, {x});
    MLK_CHECK(e1.has_value());
    b.output(*e1);
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("test");
    mlk::EGraphConfig cfg;
    cfg.maxNodes = 8;  // tiny budget
    mlk::EGraph eg(profile, cfg);
    MLK_CHECK(eg.importGraph(b.graph(), b.graph().outputs()[0]).has_value());
    for (int i = 0; i < 100; ++i) {
        auto r = eg.saturateOnce(symbols);
        MLK_CHECK(r.has_value());
        if (eg.saturated()) break;
    }
    MLK_CHECK(eg.numENodes() <= 64);
}

MLK_TEST_MAIN("egraph")
