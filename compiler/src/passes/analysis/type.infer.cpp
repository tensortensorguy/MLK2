// type.infer — abstract type inference pass (spec §8.1).
#include "../passes_common.h"
#include "mlk/type/type_inference.h"

namespace mlk::passes {

namespace {
constexpr Tier kAllTiers[] = {Tier::Tier0, Tier::Tier1, Tier::Tier2,
                              Tier::Tier3};
}  // namespace

class TypeInferPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        MLK_TRY_VAR(changed,
                    inferTypesReported(graph, *ctx.domainProfile, *ctx.diag));
        r.changed = changed != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_type_infer_pass(SymbolTable& symbols) {
    static TypeInferPass pass(symbols, "type.infer", PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"ir.verified"},
                 {"type.inferred"}, {},
                 {kAllTiers[0], kAllTiers[1], kAllTiers[2], kAllTiers[3]});
}

}  // namespace mlk::passes
