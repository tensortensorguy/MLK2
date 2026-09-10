// alias.infer — may-not-alias fact inference (spec §8.1).
#include "../passes_common.h"

namespace mlk::passes {

class AliasInferPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        (void)graph;
        // MVP alias fact: distinct placeholder values are non-aliasing;
        // node results are fresh allocations (SSA-like IR), so may-not-alias
        // holds pairwise except through explicit LayoutTransform views.
        r.changed = false;
        return r;
    }
};

void register_alias_infer_pass(SymbolTable& symbols) {
    static AliasInferPass pass(symbols, "alias.infer", PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"effect.inferred"},
                 {"alias.inferred"}, {},
                 {Tier::Tier1, Tier::Tier2, Tier::Tier3});
}

}  // namespace mlk::passes
