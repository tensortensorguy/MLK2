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

// --- Regression harness -----------------------------------------------------
//
// Minimal recursive MathGraph evaluator for extraction regressions: the
// identity-union bug produced WRONG extractions (0.0 + x -> add(0,0) = 0,
// x * 1.0 -> mul(x,x) = x^2) that no "does not error" test could catch.

namespace {

/// Recursively evaluates `v` in `g`; env maps interned names to values.
[[nodiscard]] double evalValue(const mlk::MathGraph& g, mlk::ValueId v,
                               [[maybe_unused]] const mlk::SymbolTable& symbols,
                               const mlk::SmallVector<mlk::SymbolId, 4>& names,
                               const mlk::SmallVector<double, 4>& env,
                               mlk::SmallVector<double, 16>& memo,
                               uint32_t depth) {
    MLK_CHECK(depth < 64);  // extraction graphs are tiny; no runaway
    if (v < memo.size() && memo[v] == memo[v]) return memo[v];
    const mlk::Value& val = g.value(v);
    double r = 0.0;
    switch (val.kind) {  // Rule 78: exhaustive
        case mlk::ValueKind::Constant:
            r = val.constant.isInt ? static_cast<double>(val.constant.i64)
                                   : val.constant.f64;
            break;
        case mlk::ValueKind::Variable:
        case mlk::ValueKind::Placeholder:
        case mlk::ValueKind::Symbol: {
            bool found = false;
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (names[i] == val.name) {
                    r = env[i];
                    found = true;
                }
            }
            MLK_CHECK(found);
            break;
        }
        case mlk::ValueKind::NodeResult: {
            const mlk::Node& n = g.node(val.producer);
            MLK_CHECK(n.inputs.size() <= 2);
            const double a = evalValue(g, n.inputs[0], symbols, names, env,
                                       memo, depth + 1);
            const double b2 = n.inputs.size() > 1
                                  ? evalValue(g, n.inputs[1], symbols, names,
                                              env, memo, depth + 1)
                                  : 0.0;
            switch (n.op) {  // evaluator supports the ops the tests build
                case mlk::MathOp::Add: r = a + b2; break;
                case mlk::MathOp::Sub: r = a - b2; break;
                case mlk::MathOp::Mul: r = a * b2; break;
                case mlk::MathOp::Div: r = a / b2; break;
                default: MLK_CHECK(false); r = 0.0; break;
            }
            break;
        }
    }
    if (v >= memo.size()) memo.resize(v + 1, 0.0);
    memo[v] = r;
    return r;
}

}  // namespace

MLK_TEST(egraph, identity_rules_extract_operand_not_constant) {
    // Regression: the identity rules used to union the two CHILD classes
    // (constant <-> operand), so 0.0 + x extracted as add(0,0) = 0 and
    // x * 1.0 as mul(x,x) = x^2. The PARENT class must merge with the
    // surviving child instead; the cheapest form is then the bare operand.
    mlk::SymbolTable symbols;
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("test");
    profile.numeric.ieee754 = true;

    const mlk::SymbolId xName = symbols.intern("x");
    mlk::SmallVector<mlk::SymbolId, 4> names;
    names.push_back(xName);
    mlk::SmallVector<double, 4> env;
    env.push_back(5.0);

    for (const int which : {0, 1}) {
        mlk::GraphBuilder b(symbols);
        const mlk::MathType f64 =
            mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
        const mlk::ValueId x = b.placeholder("x", f64);
        const mlk::ValueId c = b.constant(which == 0 ? 0.0 : 1.0, f64);
        auto root = which == 0 ? b.op(mlk::MathOp::Add, {c, x})
                               : b.op(mlk::MathOp::Mul, {x, c});
        MLK_CHECK(root.has_value());
        b.output(*root);
        mlk::EGraph eg(profile);
        auto rc = eg.importGraph(b.graph(), b.graph().outputs()[0]);
        MLK_CHECK(rc.has_value());
        MLK_CHECK(eg.saturateOnce(symbols).has_value());
        mlk::MathGraph out{&symbols};
        auto extracted = eg.extract(symbols, out, *rc);
        MLK_CHECK(extracted.has_value());
        // The merged root class contains the leaf x (cost 0) and the
        // add/mul node (cost >= 1): extraction must return x alone.
        MLK_CHECK_EQ(out.numValues(), 1u);
        if (out.numValues() == 1) {
            MLK_CHECK(out.values()[0].kind == mlk::ValueKind::Placeholder);
            MLK_CHECK(out.values()[0].name == xName);
            mlk::SmallVector<double, 16> memo;
            const double got = evalValue(out, *extracted, symbols, names, env,
                                         memo, 0);
            MLK_CHECK_EQ(got, 5.0);  // was 0.0 / 25.0 on the buggy union
        }
    }
}

MLK_TEST(egraph, extract_keeps_dfs_interleaved_leaves_wired) {
    // Regression: extraction materializes in DFS order, so leaves and node
    // results INTERLEAVE ((x*2)+y allocates x, 2, mul, y, add). The old
    // transplant assumed "leaves first, then results" and rewired y to the
    // mul's transplanted result (evaluating (x*2)+(x*2)).
    mlk::SymbolTable symbols;
    const mlk::SymbolId xName = symbols.intern("x");
    const mlk::SymbolId yName = symbols.intern("y");
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("test");
    profile.numeric.ieee754 = true;

    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    const mlk::ValueId y = b.placeholder("y", f64);
    const mlk::ValueId two = b.constant(2.0, f64);
    auto m = b.op(mlk::MathOp::Mul, {x, two});
    MLK_CHECK(m.has_value());
    auto a = b.op(mlk::MathOp::Add, {*m, y});
    MLK_CHECK(a.has_value());
    b.output(*a);

    mlk::EGraph eg(profile);
    auto rc = eg.importGraph(b.graph(), b.graph().outputs()[0]);
    MLK_CHECK(rc.has_value());
    // No saturation needed: the extraction must reproduce the input tree.
    mlk::MathGraph out{&symbols};
    auto extracted = eg.extract(symbols, out, *rc);
    MLK_CHECK(extracted.has_value());

    // Numeric ground truth: x=3, y=4 -> (3*2)+4 = 10 (was 12 on the bug).
    mlk::SmallVector<mlk::SymbolId, 4> names;
    names.push_back(xName);
    names.push_back(yName);
    mlk::SmallVector<double, 4> env;
    env.push_back(3.0);
    env.push_back(4.0);
    mlk::SmallVector<double, 16> memo;
    const double got =
        evalValue(out, *extracted, symbols, names, env, memo, 0);
    MLK_CHECK_EQ(got, 10.0);

    // Structural: the add's second operand is still the y LEAF.
    const mlk::Value& rootV = out.value(*extracted);
    MLK_CHECK(rootV.kind == mlk::ValueKind::NodeResult);
    if (rootV.kind == mlk::ValueKind::NodeResult) {
        const mlk::Node& rootN = out.node(rootV.producer);
        MLK_CHECK_EQ(rootN.op, mlk::MathOp::Add);
        MLK_CHECK_EQ(rootN.inputs.size(), 2u);
        const mlk::Value& rhs = out.value(rootN.inputs[1]);
        MLK_CHECK(rhs.kind == mlk::ValueKind::Placeholder);
        MLK_CHECK(rhs.name == yName);
    }
}

MLK_TEST_MAIN("egraph")
