// MLK+ Tier 0 fallback / deoptimization (Rules 4, 100-104: fallback must
// reconstruct the exact Tier 0 state; Rule 139: compiler bugs degrade, never
// crash).
#pragma once

#include "mlk/core/result.h"
#include "mlk/ir/math_graph.h"
#include "mlk/runtime/graph_state.h"
#include "mlk/runtime/telemetry.h"

namespace mlk {

/// Materialized evaluation state for the reference interpreter.
struct InterpreterState {
    /// Value -> evaluated double (scalar MVP; tensors materialize via
    /// Buffer in the executor).
    OpenHashMap<ValueId, double> scalars{};
    NodeId resumeNode{constants::kInvalidId};
    uint64_t graphVersion{0};
};

/// FallbackEngine: rebuilds InterpreterState from a GraphState snapshot.
/// Rule 102: the rebuilt state must be observationally indistinguishable
/// from the state the lower tier would have reached — verified by
/// tests/fallback.
class FallbackEngine {
public:
    FallbackEngine(TelemetrySink& telemetry, SymbolTable& symbols)
        : telemetry_(telemetry), symbols_(symbols) {}

    /// Captures a GraphState from a live interpreter state (Rule 5).
    [[nodiscard]] GraphState capture(const InterpreterState& state) const;

    /// Reconstructs interpreter state at guard failure (Rule 102).
    [[nodiscard]] Result<InterpreterState> reconstruct(
        const GraphState& snapshot) const;

    /// Rule 103: repeated fallback at the same site must be throttled.
    void recordFallback(SymbolId site);
    [[nodiscard]] bool siteThrottled(SymbolId site) const;

private:
    TelemetrySink& telemetry_;
    SymbolTable& symbols_;
    OpenHashMap<SymbolId, uint32_t> fallbackCounts_{};
};

}  // namespace mlk
