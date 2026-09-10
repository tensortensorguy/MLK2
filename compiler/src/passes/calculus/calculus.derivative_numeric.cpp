// calculus.derivative_numeric — finite-difference strategy attachment
// (spec §8.4; Rule 34: numeric differentiation REQUIRES an explicit
// accuracy contract permitting approximation).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kCalcTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class DerivativeNumericPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        // Rule 34: finite differences REQUIRE an explicit accuracy contract;
        // without one this pass must not fire.
        if (!ctx.accuracy->permitsApproximation()) {
            return err(ErrorCode::AccuracyViolation,
                       "numeric differentiation requires an accuracy "
                       "contract permitting approximation (Rule 34)", 34);
        }
        r.changed = false;  // strategy attached at Tier-2 in autotuner
        return r;
    }
};

void register_calculus_derivative_numeric_pass(SymbolTable& symbols) {
    static DerivativeNumericPass pass(symbols, "calculus.derivative_numeric",
                                      PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering, {"accuracy.analyzed"},
                 {"calculus.strategy"}, {},
                 {kCalcTiers[0], kCalcTiers[1], kCalcTiers[2]});
}

}  // namespace mlk::passes
