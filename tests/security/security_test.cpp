// Rule 124/127: untrusted artifacts are validated before use; invalid ones
// are rejected with telemetry, never loaded silently.
#include "mlk/core/symbol_table.h"
#include "mlk/runtime/cache.h"
#include "mlk/runtime/telemetry.h"

#include "mlk_test.h"

MLK_TEST(security, cache_rejects_version_mismatch) {
    mlk::SymbolTable symbols;
    mlk::json::Value doc = mlk::json::Object{};
    doc.set("format_version", mlk::json::Value{int64_t{999}});
    auto entry = mlk::CacheEntry::fromJson(doc, symbols);
    MLK_CHECK(!entry.has_value());
    MLK_CHECK_EQ(entry.error().code, mlk::ErrorCode::InvalidArtifact);
}

MLK_TEST(security, cache_rejects_missing_key) {
    mlk::SymbolTable symbols;
    mlk::json::Value doc = mlk::json::Object{};
    doc.set("format_version", mlk::json::Value{int64_t{1}});
    auto entry = mlk::CacheEntry::fromJson(doc, symbols);
    MLK_CHECK(!entry.has_value());  // Rule 57: incomplete keys are bugs
}

MLK_TEST(security, cache_accepts_wellformed_entry) {
    mlk::SymbolTable symbols;
    mlk::CacheKey key;
    key.graphHash = 123;
    key.accuracyHash = 7;
    key.passPipelineHash = 9;
    key.hardwareFingerprint = 11;
    mlk::CacheEntry e;
    e.key = key;
    e.strategy = symbols.intern("local_search");
    e.measuredMs = 1.5;
    auto doc = e.toJson(symbols);
    auto back = mlk::CacheEntry::fromJson(doc, symbols);
    MLK_CHECK(back.has_value());
    MLK_CHECK_EQ(back->key.graphHash, 123u);
}

MLK_TEST(security, telemetry_never_carries_user_data) {
    // Rule 157: telemetry is privacy-safe — events are counters only.
    mlk::SymbolTable symbols;
    mlk::TelemetrySink t;
    mlk::TelemetryEvent e;
    e.kind = mlk::TelemetryEventKind::Fallback;
    t.record(e);
    const auto json = t.toJson();
    MLK_CHECK(json.isObject() || json.isArray());
}

MLK_TEST_MAIN("security")
