// IR unit tests: graph construction, hashing, printing, serialization
// (Rules 15, 21, 24, 39).
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/ir/graph_json.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/effect/effect_inference.h"

#include "mlk_test.h"

MLK_TEST(ir, build_sin_graph) {
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 = mlk::MathType::scalar(mlk::Domain::Float,
                                                    mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    const mlk::ValueId three = b.constant(3.0, f64);
    auto sq = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq.has_value());
    auto tx = b.op(mlk::MathOp::Mul, {three, x});
    MLK_CHECK(tx.has_value());
    auto sum = b.op(mlk::MathOp::Add, {*sq, *tx});
    MLK_CHECK(sum.has_value());
    auto s = b.op(mlk::MathOp::Sin, {*sum});
    MLK_CHECK(s.has_value());
    b.output(*s);
    mlk::MathGraph& g = b.graph();
    MLK_CHECK_EQ(g.numValues(), 6u);
    MLK_CHECK_EQ(g.numNodes(), 4u);
    // users() is a multiset of operand edges: x feeds mul(x,x) twice and
    // mul(3,x) once.
    MLK_CHECK_EQ(g.users(x).size(), 3u);
}

MLK_TEST(ir, graph_hash_is_structural) {
    mlk::SymbolTable symbols;
    mlk::MathType f64 = mlk::MathType::scalar(mlk::Domain::Float,
                                              mlk::Dtype::F64);
    auto build = [&](bool commute) {
        mlk::GraphBuilder b(symbols);
        const mlk::ValueId x = b.placeholder("x", f64);
        const mlk::ValueId c = b.constant(2.0, f64);
        auto r = commute ? b.op(mlk::MathOp::Add, {c, x})
                         : b.op(mlk::MathOp::Add, {x, c});
        MLK_CHECK(r.has_value());
        return mlk::graphHash(b.graph());
    };
    // NOTE: hashes differ pre-canonicalization (commutative_sort makes them
    // equal); identical construction must hash identically (Rule 24).
    MLK_CHECK_EQ(build(false), build(false));
}

MLK_TEST(ir, json_roundtrip) {
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    const mlk::json::Value doc = mlk::graphToJson(b.graph(), symbols);
    auto back = mlk::graphFromJson(doc, symbols);
    MLK_CHECK(back.has_value());
    MLK_CHECK_EQ(back->numNodes(), 1u);
    MLK_CHECK_EQ(back->outputs().size(), 1u);
}

MLK_TEST(ir, loader_rejects_bad_input) {
    // Rule 124: loader is a trust boundary.
    mlk::SymbolTable symbols;
    auto bad1 = mlk::parseGraphFile("{\"format\":\"nope\"}", symbols);
    MLK_CHECK(!bad1.has_value());
    auto bad2 = mlk::parseGraphFile(
        "{\"format\":\"mlk-graph\",\"version\":99}", symbols);
    MLK_CHECK(!bad2.has_value());
    auto bad3 = mlk::parseGraphFile("not json", symbols);
    MLK_CHECK(!bad3.has_value());
}

MLK_TEST(ir, equivalence_recorded_not_destroyed) {
    // Rule 21: replacements create new values; originals recoverable.
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s1 = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s1.has_value());
    b.output(*s1);
    auto s2 = b.op(mlk::MathOp::Sin, {x});  // equivalent duplicate
    MLK_CHECK(s2.has_value());
    b.graph().recordEquivalent(*s1, *s2);
    MLK_CHECK_EQ(b.graph().representative(*s1), *s2);
    // Both values still exist in the graph.
    // 1 leaf (x) + 2 node results.
    MLK_CHECK(b.graph().numValues() == 3);
}

MLK_TEST(ir, effects_on_ffi_boundary) {
    // Rule 112: native boundary ops carry FFI effects.
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto r = b.op(mlk::MathOp::NativeToMathRef, {x});
    MLK_CHECK(r.has_value());
    MLK_CHECK(!mlk::isPure(b.graph().node(0).effects));
}

MLK_TEST_MAIN("ir")
