// MLK+ cost model (cost/cost_model.h) + roofline (Rule 55: cost models must
// include lower bounds).
#pragma once

#include <cstdint>

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/ir/math_graph.h"

namespace mlk {

/// Hardware description (Rule 31: no assumption of stable hardware — values
/// are queried/probed at runtime, never hardcoded).
struct HardwareInfo {
    std::size_t cacheLineBytes{constants::kDefaultCacheLineBytes};
    std::size_t simdWidthF32{constants::kDefaultSimdWidthF32};
    uint32_t cores{1};
    double peakFlopsPerCycleF32{
        constants::kDefaultPeakFlopsPerCorePerCycleF32};
    double peakFlopsPerCycleF64{
        constants::kDefaultPeakFlopsPerCorePerCycleF64};
    double memoryBandwidthGiBs{constants::kDefaultMemoryBandwidthGiBs};
    double clockGHz{2.5};
    bool hasFMA{true};

    /// Runtime probe (x86 features via builtins; conservative fallbacks).
    [[nodiscard]] static HardwareInfo detect();
    [[nodiscard]] HashValue fingerprint() const noexcept;
};

/// Static cost estimate of a node/graph (Rule 55 fields).
struct CostEstimate {
    double flops{0.0};
    double bytesMoved{0.0};
    double instructions{0.0};
    double latencyNs{0.0};

    [[nodiscard]] double arithmeticIntensity() const {
        return bytesMoved > 0.0 ? flops / bytesMoved : 0.0;
    }
};

class CostModel {
public:
    virtual ~CostModel() = default;
    [[nodiscard]] virtual CostEstimate nodeCost(const MathGraph& graph,
                                                NodeId node,
                                                const HardwareInfo& hw) = 0;
    [[nodiscard]] virtual CostEstimate graphCost(const MathGraph& graph,
                                                 const HardwareInfo& hw);
};

/// Table-driven basic cost model (op costs are named constants; Rule 27).
class BasicCostModel final : public CostModel {
public:
    [[nodiscard]] CostEstimate nodeCost(const MathGraph& graph, NodeId node,
                                        const HardwareInfo& hw) override;
};

/// Roofline lower bound (spec §14.4):
///   time >= max(flops / peak_flops, bytes / memory_bandwidth)
[[nodiscard]] double rooflineLowerBoundNs(const CostEstimate& c,
                                          const HardwareInfo& hw);

/// Roofline lower bound under CANDIDATE resource limits (Rule 55: a
/// config-aware bound makes pruning sound). A candidate that uses
/// `threads` of the machine's cores cannot exceed that fraction of the
/// all-core peak — neither for flops nor for memory bandwidth — and a
/// candidate whose vector width is below the machine SIMD width cannot
/// exceed that fraction of the flop peak. Both caps can only LOWER the
/// achievable throughput, so the resulting bound stays a valid lower
/// bound for that candidate. threads <= 0 / vectorWidth <= 0 (unset
/// knobs) fall back to the machine-wide bound.
[[nodiscard]] double rooflineCandidateLowerBoundNs(const CostEstimate& c,
                                                   const HardwareInfo& hw,
                                                   int64_t threads,
                                                   int64_t vectorWidth);

}  // namespace mlk
