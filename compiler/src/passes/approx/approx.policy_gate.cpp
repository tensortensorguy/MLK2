// approx.policy_gate — the single authority on approximation legality
// (spec §8.6; Rule 34/91: no approximation without an explicit, verified
// contract). Downstream approximation passes consult this decision.
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kApproxTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolicyGatePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // The gate is the single authority (spec §8.6): downstream
        // approximation passes check this recorded decision.
        const bool allowed =
            ctx.domainProfile->has(Capability::HasApproximation) &&
            ctx.domainProfile->approximation.allowApproximation &&
            (ctx.accuracy != nullptr && ctx.accuracy->permitsApproximation());
        Fact f;
        f.property = PropertyId::Pure;
        f.value = allowed ? TriState::Unknown : TriState::True;
        f.period = allowed ? 1.0 : 0.0;  // payload: 1 = approx permitted
        for (const ValueId vid : graph.outputs()) {
            graph.value(vid).facts.set(f);
        }
        if (!allowed) {
            Diagnostic d;
            d.severity = Severity::Note;
            d.message =
                "approximation policy gate: approximation disabled by "
                "profile/contract (Rule 34)";
            ctx.diag->report(d);
        }
        r.changed = false;
        return r;
    }
};

void register_approx_policy_gate_pass(SymbolTable& symbols) {
    static PolicyGatePass pass(symbols, "approx.policy_gate",
                               PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"accuracy.analyzed"},
                 {"approx.gated"}, {}, {kApproxTiers[0], kApproxTiers[1]});
}

}  // namespace mlk::passes
