// MLK+ benchmark harness contract (Rule 49: statistically valid measurement;
// Rule 56: real workloads — seeded synthetic data with realistic structure,
// never all-zero buffers).
#pragma once

#include "mlk/autotune/candidate.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"

namespace mlk {

struct BenchmarkProtocol {
    uint32_t warmup{constants::kBenchDefaultWarmup};
    uint32_t reps{constants::kBenchDefaultReps};
};

/// Executes the graph over seeded, realistic inputs and measures with
/// min/median/stddev selection + noise detection (Rule 59).
[[nodiscard]] Result<BenchmarkMeasurement> benchmarkGraph(
    const MathGraph& graph, SymbolTable& symbols, const BenchmarkProtocol& p);

}  // namespace mlk
