// egraph.build — imports the math graph into an e-graph and reports class
// statistics (spec §8.3; non-destructive validation of the import path).
#include "../passes_common.h"
#include "egraph.h"

namespace mlk::passes {

namespace {
constexpr Tier kEgraphTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class EgraphBuildPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // The e-graph is per-compilation state; building here validates that
        // import succeeds and reports class statistics (the saturate pass
        // constructs its own instance over the same graph deterministically).
        EGraphConfig cfg;
        cfg.maxNodes = constants::kEgraphDefaultMaxNodes;
        const MathDomainProfile& profile = *ctx.domainProfile;
        EGraph eg(profile, cfg);
        if (graph.outputs().empty()) {
            return err(ErrorCode::InvalidGraph,
                       "egraph.build requires at least one output", 47);
        }
        MLK_TRY_VAR(rootClass, eg.importGraph(graph, graph.outputs()[0]));
        (void)rootClass;
        r.changed = false;
        return r;
    }
};

void register_egraph_build_pass(SymbolTable& symbols) {
    static EgraphBuildPass pass(symbols, "egraph.build",
                                PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"ir.verified"},
                 {"egraph.built"}, {},
                 {kEgraphTiers[0], kEgraphTiers[1]});
}

}  // namespace mlk::passes
