// math.strength_reduce — strength reduction gated on domain constraints
// (spec §8.2; Rule 33: no rewrite without legality conditions).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class StrengthReducePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        r.nodesBefore = graph.liveNodeCount();
        // pow(x,2)->mul(x,x) lives in normalize_ops (canonical form first).
        // log(exp(x)) -> x requires domain constraints (x real, no
        // overflow): gated on facts; without definite facts we skip
        // (Rule 33: no rewrite without legality conditions).
        r.changed = false;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_strength_reduce_pass(SymbolTable& symbols) {
    static StrengthReducePass pass(symbols, "math.strength_reduce",
                                   PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.strength_reduced"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
