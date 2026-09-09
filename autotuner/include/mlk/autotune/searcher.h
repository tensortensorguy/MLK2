// MLK+ autotuner searcher (realization spec §15): seed heuristics → local
// search, correctness-first (Rule 58), noise-filtered (Rule 59).
#pragma once

#include "mlk/autotune/benchmarker.h"
#include "mlk/autotune/candidate.h"
#include "mlk/autotune/search_space.h"
#include "mlk/runtime/cache.h"
#include "mlk/runtime/telemetry.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

struct TuningContext {
    const MathDomainProfile* profile{nullptr};
    const AccuracyContract* accuracy{nullptr};
    HardwareInfo hardware{};
    uint32_t maxCandidates{constants::kTuneDefaultMaxCandidates};
    BenchmarkProtocol protocol{};
};

class Autotuner {
public:
    Autotuner(SymbolTable& symbols, TelemetrySink& telemetry);

    [[nodiscard]] Result<CacheEntry> tune(const MathGraph& graph,
                                          const TuningContext& ctx,
                                          RealizationCache& cache);

    [[nodiscard]] const SearchSpace& lastSpace() const { return space_; }

private:
    [[nodiscard]] SearchSpace buildSpace(const MathGraph& graph) const;
    [[nodiscard]] Result<bool> verifyCandidate(const MathGraph& graph,
                                               const TuningContext& ctx,
                                               const TuningCandidate& c) const;
    [[nodiscard]] Result<BenchmarkMeasurement> measureCandidate(
        const MathGraph& graph, const TuningContext& ctx,
        const TuningCandidate& c) const;

    SymbolTable& symbols_;
    TelemetrySink& telemetry_;
    SearchSpace space_{};
};

}  // namespace mlk
