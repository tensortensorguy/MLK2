// Rule 102/41.5: guard failure must reconstruct the EXACT lower-tier state.
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/runtime/fallback.h"
#include "mlk/runtime/telemetry.h"

#include "mlk_test.h"

MLK_TEST(fallback, guard_failure_reconstructs_exact_state) {
    mlk::SymbolTable symbols;
    mlk::TelemetrySink telemetry;
    mlk::FallbackEngine engine(telemetry, symbols);

    // Simulated Tier-0 state before speculation.
    mlk::InterpreterState pre;
    pre.resumeNode = 4;
    pre.graphVersion = 17;
    double* a = pre.scalars.findOrInsert(1, nullptr, 2.0);
    *a = 2.0;
    double* b = pre.scalars.findOrInsert(5, nullptr, -0.7568024953079282);
    *b = -0.7568024953079282;

    // Speculative kernel runs; guard fails; deopt via GraphState.
    const mlk::GraphState snap = engine.capture(pre);
    auto recovered = engine.reconstruct(snap);
    MLK_CHECK(recovered.has_value());
    // Observationally indistinguishable (Rule 102):
    MLK_CHECK_EQ(recovered->resumeNode, pre.resumeNode);
    MLK_CHECK_EQ(recovered->graphVersion, pre.graphVersion);
    pre.scalars.forEach([&](mlk::ValueId id, double value) {
        const double* v = recovered->scalars.find(id);
        MLK_CHECK(v != nullptr);
        if (v != nullptr) MLK_CHECK_EQ(*v, value);
    });
    // Telemetry recorded (Rule 30).
    engine.recordFallback(symbols.intern("test_site"));
    MLK_CHECK(telemetry.countKind(mlk::TelemetryEventKind::Fallback) >= 1);
}

MLK_TEST(fallback, throttle_downgrades_repeated_failures) {
    // Rule 103: repeated fallback at the same site disables the speculation.
    mlk::SymbolTable symbols;
    mlk::TelemetrySink telemetry;
    mlk::FallbackEngine engine(telemetry, symbols);
    const mlk::SymbolId site = symbols.intern("site_x");
    for (uint32_t i = 0; i < 64; ++i) engine.recordFallback(site);
    MLK_CHECK(engine.siteThrottled(site));
}

MLK_TEST_MAIN("fallback")
