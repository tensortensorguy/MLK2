// accuracy.analyze — records the effective accuracy contract as a graph
// fact; gates all approximation passes (spec §8.1, Rule 34).
#include "../passes_common.h"

namespace mlk::passes {

class AccuracyAnalyzePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Record the effective contract as a graph-level fact: every
        // approximation decision downstream must consult it (Rule 34).
        const bool approxAllowed =
            ctx.domainProfile->has(Capability::HasApproximation) &&
            ctx.accuracy->permitsApproximation();
        for (const ValueId vid : graph.outputs()) {
            Value& v = graph.value(vid);
            v.facts.setTriState(
                PropertyId::Pure,
                approxAllowed ? TriState::Unknown : TriState::True);
        }
        r.changed = false;  // analysis pass: no IR change (Rule 10)
        return r;
    }
};

void register_accuracy_analyze_pass(SymbolTable& symbols) {
    static AccuracyAnalyzePass pass(symbols, "accuracy.analyze",
                                    PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"property.inferred"},
                 {"accuracy.analyzed"}, {},
                 {Tier::Tier1, Tier::Tier2, Tier::Tier3});
}

}  // namespace mlk::passes
