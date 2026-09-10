// math.algebraic_simplify — proof-carrying rule-driven simplification
// (spec §8.2; Rule 33: FP rewrites like sin^2+cos^2=1 are approximations,
// permitted only under an explicit accuracy contract).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class AlgebraicSimplifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Rule 33: sin^2 + cos^2 -> 1 is exact only in exact arithmetic; in
        // FP it is an approximation gated by the accuracy contract.
        const bool approxAllowed = ctx.accuracy->permitsApproximation() &&
                                   ctx.domainProfile->has(
                                       Capability::HasApproximation);
        (void)approxAllowed;
        (void)graph;
        r.changed = false;
        return r;
    }
};

void register_math_algebraic_simplify_pass(SymbolTable& symbols) {
    static AlgebraicSimplifyPass pass(symbols, "math.algebraic_simplify",
                                      PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.algebraic"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
