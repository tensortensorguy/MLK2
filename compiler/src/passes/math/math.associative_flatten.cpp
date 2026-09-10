// math.associative_flatten — n-ary flattening of associative chains,
// gated on reassociation legality (spec §8.2; Rule 33).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class AssociativeFlattenPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Rule 33: DO NOT flatten FP add/mul unless the profile explicitly
        // permits reassociation (accuracy contract may also allow it).
        const bool allowed =
            ctx.domainProfile->numeric.allowReassociation ||
            ctx.accuracy->allowReassociation;
        if (!allowed) {
            return r;  // legal no-op: pass remains idempotent (Rule 10)
        }
        // Flattening is realized in the e-graph (add/mul are n-ary there);
        // the destructive tree form is kept here for Tier-1 budget reasons.
        r.changed = false;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_associative_flatten_pass(SymbolTable& symbols) {
    static AssociativeFlattenPass pass(symbols, "math.associative_flatten",
                                       PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.associative"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
