// Runtime tests: GraphState, fallback reconstruction (Rule 102), telemetry.
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/fallback.h"
#include "mlk/runtime/telemetry.h"

#include "mlk_test.h"

MLK_TEST(runtime, graph_state_roundtrip) {
    mlk::SymbolTable symbols;
    mlk::TelemetrySink telemetry;
    mlk::FallbackEngine engine(telemetry, symbols);
    mlk::InterpreterState state;
    state.resumeNode = 3;
    state.graphVersion = 42;
    double* a = state.scalars.findOrInsert(7, nullptr, 3.5);
    *a = 3.5;
    const mlk::GraphState snap = engine.capture(state);
    MLK_CHECK_EQ(snap.resumeNode, 3u);
    MLK_CHECK_EQ(snap.graphVersion, 42u);
    auto rebuilt = engine.reconstruct(snap);
    MLK_CHECK(rebuilt.has_value());
    const double* v = rebuilt->scalars.find(7);
    MLK_CHECK(v != nullptr);
    MLK_CHECK_NEAR(*v, 3.5, 1e-15);
}

MLK_TEST(runtime, graph_state_json_roundtrip) {
    mlk::GraphState s;
    s.resumeNode = 5;
    s.graphVersion = 9;
    mlk::ValueBinding b;
    b.value = 11;
    b.f64 = 2.25;
    s.bindings.push_back(b);
    const auto doc = s.toJson();
    auto back = mlk::GraphState::fromJson(doc);
    MLK_CHECK(back.has_value());
    MLK_CHECK_EQ(back->resumeNode, 5u);
    MLK_CHECK_EQ(back->bindings.size(), 1u);
    MLK_CHECK_NEAR(back->bindings[0].f64, 2.25, 1e-15);
}

MLK_TEST(runtime, graph_state_rejects_incomplete) {
    // Rule 101: the verifier rejects incomplete GraphState.
    mlk::json::Value doc = mlk::json::Object{};
    doc.set("resume_node", mlk::json::Value{static_cast<int64_t>(7)});
    doc.set("bindings", mlk::json::Array{});
    auto r = mlk::GraphState::fromJson(doc);
    MLK_CHECK(!r.has_value());
}

MLK_TEST(runtime, fallback_site_throttling) {
    // Rule 103: repeated fallbacks at the same site are throttled.
    mlk::SymbolTable symbols;
    mlk::TelemetrySink telemetry;
    mlk::FallbackEngine engine(telemetry, symbols);
    const mlk::SymbolId site = symbols.intern("kernel_exec_site_a");
    for (uint32_t i = 0; i < 32; ++i) engine.recordFallback(site);
    MLK_CHECK(engine.siteThrottled(site));
    const mlk::SymbolId other = symbols.intern("kernel_exec_site_b");
    engine.recordFallback(other);
    MLK_CHECK(!engine.siteThrottled(other));
}

MLK_TEST(runtime, telemetry_records_fallback_events) {
    // Rule 30: no silent fallbacks.
    mlk::SymbolTable symbols;
    mlk::TelemetrySink telemetry;
    mlk::FallbackEngine engine(telemetry, symbols);
    engine.recordFallback(symbols.intern("site"));
    MLK_CHECK(telemetry.countKind(mlk::TelemetryEventKind::Fallback) == 1);
    const auto json = telemetry.toJson();
    MLK_CHECK(json.isObject() || json.isArray());
}

MLK_TEST(runtime, tier0_interpreter_matches_reference_math) {
    mlk::SymbolTable symbols;
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
    mlk::SmallVector<double, 8> inputs{2.0};
    auto out = mlk::interpretGraph(b.graph(), inputs);
    MLK_CHECK(out.has_value());
    // sin(4+6) = -0.5440211108893698 (Tier 0 is authoritative; Rule 85)
    MLK_CHECK_NEAR(out->outputScalars[0], -0.5440211108893698, 1e-15);
}

MLK_TEST_MAIN("runtime")
