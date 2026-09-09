// MLK+ tuning candidate (spec §15 RealizationCandidate).
#pragma once

#include "mlk/core/small_vector.h"
#include "mlk/cost/cost_model.h"

namespace mlk {

struct BenchmarkMeasurement {
    double minMs{0.0};
    double medianMs{0.0};
    double stddevMs{0.0};
    double opsPerSecond{0.0};
    uint32_t reps{0};
};

struct TuningCandidate {
    SmallVector<int64_t, 8> config{};
    std::optional<BenchmarkMeasurement> estimate;
    std::optional<BenchmarkMeasurement> measured;
    bool verified{false};          // Rule 58: correctness precedes benchmarking
    double costLowerBoundNs{0.0};  // Rule 55: prune by lower bound
};

}  // namespace mlk
