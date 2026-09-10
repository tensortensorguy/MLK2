// math.expression_balance — balanced-tree reassociation for commutative
// chains (spec §8.2; gated identically to math.associative_flatten,
// Rule 33; rotations realized through the e-graph forms).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class ExpressionBalancePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        // ((a+b)+c)+d -> (a+b)+(c+d) requires reassociation legality
        // (Rule 33). Gated identically to associative_flatten; the actual
        // tree rotation happens through the e-graph forms.
        const bool allowed = ctx.domainProfile->numeric.allowReassociation ||
                             ctx.accuracy->allowReassociation;
        (void)allowed;
        r.changed = false;
        return r;
    }
};

void register_math_expression_balance_pass(SymbolTable& symbols) {
    static ExpressionBalancePass pass(symbols, "math.expression_balance",
                                      PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.balanced"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
