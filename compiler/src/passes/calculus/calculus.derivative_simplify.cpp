// calculus.derivative_simplify — canonical cleanup of derivative results
// (spec §8.4: 2x sin(x) + x² cos(x) balanced form is e-graph territory;
// here fold/sort/identity elimination run over the lowered derivative).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kCalcTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class DerivativeSimplifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Runs the canonicalize driver over derivative results (spec:
        // 2x sin(x) + x² cos(x) -> x(2 sin(x) + x cos(x)) form is e-graph
        // territory; here we fold/sort/eliminate identities).
        Pass* canonicalize =
            PassRegistry::instance().byName(ctx.symbols->intern("math.canonicalize"));
        if (canonicalize == nullptr) {
            return err(ErrorCode::Internal, "math.canonicalize not registered");
        }
        MLK_TRY_VAR(cr, canonicalize->run(ctx, graph));
        r.changed = cr.changed;
        return r;
    }
};

void register_calculus_derivative_simplify_pass(SymbolTable& symbols) {
    static DerivativeSimplifyPass pass(symbols,
                                       "calculus.derivative_simplify",
                                       PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"calculus.lowered"},
                 {"math.canonical"}, {"analysis.cse"},
                 {kCalcTiers[0], kCalcTiers[1], kCalcTiers[2]});
}

}  // namespace mlk::passes
