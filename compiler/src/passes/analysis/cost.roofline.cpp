// cost.roofline — roofline lower-bound analysis (spec §8.1; Rule 23: the
// bound is telemetry, never stored in the math graph).
#include "../passes_common.h"
#include "mlk/core/event_sink.h"
#include "mlk/cost/cost_model.h"

#include <cmath>

namespace mlk::passes {

class CostRooflinePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        if (ctx.costModel == nullptr) {
            return err(ErrorCode::InvalidArgument,
                       "cost.roofline requires a cost model in context",
                       55);
        }
        const HardwareInfo hw = HardwareInfo::detect();
        double totalFlops = 0.0;
        double totalBytes = 0.0;
        for (const NodeId nid : graph.topoOrder()) {
            const CostEstimate c = ctx.costModel->nodeCost(graph, nid, hw);
            totalFlops += c.flops;
            totalBytes += c.bytesMoved;
        }
        CostEstimate total;
        total.flops = totalFlops;
        total.bytesMoved = totalBytes;
        const double lowerNs = rooflineLowerBoundNs(total, hw);
        // The bound rides the SANCTIONED channel (Rule 23: telemetry, never
        // the IR): a PerfCounter event whose counter is the lower bound in
        // whole nanoseconds (saturating on overflow — bounds beyond ~584
        // years report UINT64_MAX; still a valid "astronomically large"
        // signal). Previously the bound was computed and discarded.
        if (ctx.telemetry != nullptr) {
            uint64_t bound = 0;
            if (lowerNs > 0.0) {
                // Ceil so a positive bound never truncates to 0 (scalar
                // graphs have sub-ns bounds); the 1.8e19 cap keeps the
                // double->uint64 conversion inside range (anything above
                // is reported at the cap — still "astronomically large").
                const double capped = lowerNs >= 1.8e19 ? 1.8e19 : lowerNs;
                bound = static_cast<uint64_t>(std::ceil(capped));
            }
            ctx.telemetry->event(
                11, nameId(*ctx.symbols),
                ctx.symbols->intern("roofline_lower_bound_ns"), bound);
        }
        return r;
    }
};

void register_cost_roofline_pass(SymbolTable& symbols) {
    static CostRooflinePass pass(symbols, "cost.roofline",
                                 PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"shape.inferred"},
                 {"analysis.cost"}, {},
                 {Tier::Tier1, Tier::Tier2, Tier::Tier3});
}

}  // namespace mlk::passes
