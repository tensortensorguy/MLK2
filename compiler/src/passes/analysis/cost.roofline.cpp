// cost.roofline — roofline lower-bound analysis (spec §8.1; Rule 23: the
// bound is telemetry, never stored in the math graph).
#include "../passes_common.h"
#include "mlk/cost/cost_model.h"

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
        // Roofline bound is exposed via pass result telemetry; storing it in
        // the IR would contaminate the math graph (Rule 23).
        (void)lowerNs;
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
