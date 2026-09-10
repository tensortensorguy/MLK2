// egraph.saturate — budgeted equality-saturation loop (spec §8.3; Rule 10:
// strict fixpoint budget; Rule 21: non-destructive, all forms retained).
#include "../passes_common.h"
#include "egraph.h"
#include "mlk/core/cancellation.h"

namespace mlk::passes {

namespace {
constexpr Tier kEgraphTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class EgraphSaturatePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        EGraphConfig cfg;
        cfg.maxNodes = constants::kEgraphDefaultMaxNodes;
        cfg.maxIterations = ctx.budget.fixpointIterations;
        EGraph eg(*ctx.domainProfile, cfg);
        MLK_TRY_VAR(rootClass, eg.importGraph(graph, graph.outputs()[0]));
        (void)rootClass;
        // Budgeted fixpoint (Rule 10): saturate until no growth or budget.
        for (uint32_t iter = 0; iter < cfg.maxIterations; ++iter) {
            if (ctx.cancel != nullptr && ctx.cancel->cancelled()) {
                return err(ErrorCode::Cancelled, "saturate cancelled", 132);
            }
            MLK_TRY_VAR(grew, eg.saturateOnce(*ctx.symbols));
            if (!grew) break;
        }
        r.changed = false;  // non-destructive: e-graph keeps all forms
        return r;
    }
};

void register_egraph_saturate_pass(SymbolTable& symbols) {
    static EgraphSaturatePass pass(symbols, "egraph.saturate",
                                   PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"egraph.built"},
                 {"egraph.saturated"}, {},
                 {kEgraphTiers[0], kEgraphTiers[1]});
}

}  // namespace mlk::passes
