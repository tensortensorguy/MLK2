// MLK+ runtime telemetry (Rule 30: no silent fallbacks; Rule 157:
// structured, stable, privacy-safe telemetry).
#pragma once

#include "mlk/core/constants.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/core/event_sink.h"
#include "mlk/support/json.h"

#include <mutex>
#include <vector>

namespace mlk {

enum class TelemetryEventKind : uint8_t {
    CompileAttempt,
    CompileFailure,
    TuneAttempt,
    TuneFailure,
    Fallback,
    GuardFailure,
    Invalidation,
    BudgetViolation,
    CachePressure,
    Blacklist,
    TierTransition,
    PerfCounter,
};

const char* telemetryEventName(TelemetryEventKind k) noexcept;

struct TelemetryEvent {
    TelemetryEventKind kind{TelemetryEventKind::PerfCounter};
    SymbolId pass{kInvalidSymbolId};
    SymbolId reason{kInvalidSymbolId};
    uint64_t counter{0};
    /// Tier transition payload (Rule 138).
    uint8_t fromTier{0};
    uint8_t toTier{0};
    uint64_t graphVersion{0};
};

/// Bounded ring of events. Thread-safe via mutex (cold-path writes only).
/// Rule 157: no source graphs, no user data, no secrets.
class TelemetrySink final : public IEventSink {
public:
    void event(uint32_t kind, SymbolId pass, SymbolId reason,
               uint64_t counter) noexcept override {
        TelemetryEvent e;
        e.kind = static_cast<TelemetryEventKind>(kind);
        e.pass = pass;
        e.reason = reason;
        e.counter = counter;
        record(e);
    }

    void record(TelemetryEvent e);
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] TelemetryEvent at(std::size_t i) const;
    [[nodiscard]] uint32_t countKind(TelemetryEventKind k) const;

    /// Dump as JSON lines (stable schema, schemas/telemetry.schema.json).
    [[nodiscard]] json::Value toJson() const;

private:
    mutable std::mutex mutex_;  // cold path only (Rule 137: not on hot paths)
    std::vector<TelemetryEvent> events_;
};

}  // namespace mlk
