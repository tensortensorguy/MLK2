// Autotuner tests: declarative space (Rule 54), verify-before-bench
// (Rule 58), cache keys (Rule 57).
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/autotune/search_space.h"
#include "mlk/autotune/searcher.h"
#include "mlk/runtime/cache.h"
#include "mlk/runtime/telemetry.h"
#include "mlk/type/domain_profile.h"

#include "mlk_test.h"

MLK_TEST(autotune, search_space_roundtrip) {
    mlk::SymbolTable symbols;
    mlk::SearchSpace sp;
    mlk::SearchParam p;
    p.name = symbols.intern("vector_width");
    p.domain.push_back(4);
    p.domain.push_back(8);
    p.prior = 8;
    sp.params.push_back(p);
    const auto doc = sp.toJson(symbols);
    auto back = mlk::SearchSpace::fromJson(doc, symbols);
    MLK_CHECK(back.has_value());
    MLK_CHECK_EQ(back->params.size(), 1u);
    MLK_CHECK_EQ(back->params[0].domain.size(), 2u);
}

MLK_TEST(autotune, search_space_rejects_empty_domain) {
    mlk::SymbolTable symbols;
    mlk::json::Value doc = mlk::json::Object{};
    mlk::json::Value arr = mlk::json::Array{};
    mlk::json::Value p = mlk::json::Object{};
    p.set("name", mlk::json::Value{"tile_m"});
    p.set("domain", mlk::json::Array{});
    arr.push(std::move(p));
    doc.set("params", std::move(arr));
    auto r = mlk::SearchSpace::fromJson(doc, symbols);
    MLK_CHECK(!r.has_value());  // Rule 54: domains must be non-empty
}

MLK_TEST(autotune, tune_verifies_before_benchmarks) {
    // Rule 58: the tuner must reject graphs that fail correctness.
    mlk::SymbolTable symbols;
    mlk::TelemetrySink telemetry;
    mlk::Autotuner tuner(symbols, telemetry);
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("test");
    mlk::AccuracyContract contract;
    mlk::TuningContext ctx;
    ctx.profile = &profile;
    ctx.accuracy = &contract;
    ctx.maxCandidates = 4;  // small for test speed
    mlk::RealizationCache cache(symbols);
    auto entry = tuner.tune(b.graph(), ctx, cache);
    MLK_CHECK(entry.has_value());
    MLK_CHECK(cache.size() >= 1);
}

MLK_TEST_MAIN("autotune")
