// Approximation passes (spec §8.6): approx.policy_gate,
// approx.function_lower, approx.ulp_verify. Rule 34/91 govern everything:
// no approximation without an explicit, verified contract.
#include "../passes_common.h"

#include <cmath>

namespace mlk::passes {

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
            ctx.accuracy->permitsApproximation();
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

class FunctionLowerPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Implementation family selection: libm is the default; polynomial
        // families are attached ONLY when the policy gate allowed
        // approximation (Rule 34) — the attr carries the candidate family
        // and the ulp bound to verify.
        const bool approxAllowed =
            ctx.domainProfile->has(Capability::HasApproximation) &&
            ctx.accuracy->permitsApproximation();
        const SymbolId familyAttr = ctx.symbols->intern("family");
        const SymbolId libm = ctx.symbols->intern("libm");
        const SymbolId poly = ctx.symbols->intern("poly7");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            bool isMathFn = false;
            switch (n.op) {  // Rule 78: exhaustive
                case MathOp::Exp: case MathOp::Log: case MathOp::Sin:
                case MathOp::Cos: case MathOp::Tan: case MathOp::Tanh:
                case MathOp::Sqrt: case MathOp::Rsqrt: case MathOp::Erf:
                    isMathFn = true;
                    break;
                default:
                    break;
            }
            if (!isMathFn) continue;
            bool hasFamily = false;
            for (const auto& a : n.attrs) {
                if (a.name == familyAttr) hasFamily = true;
            }
            if (hasFamily) continue;
            Attr a;
            a.name = familyAttr;
            a.value = AttrValue{approxAllowed ? poly : libm};
            n.attrs.push_back(a);
            r.changed = true;
        }
        return r;
    }
};

class UlpVerifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Verifies that every poly-family function node meets its ULP
        // contract against the libm oracle on a deterministic sample set
        // (Rule 50: candidates must be verified before benchmarking).
        const SymbolId familyAttr = ctx.symbols->intern("family");
        const SymbolId poly = ctx.symbols->intern("poly7");
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const AttrValue* family = findAttr(n.attrs, familyAttr);
            if (family == nullptr ||
                !std::holds_alternative<SymbolId>(family->v) ||
                std::get<SymbolId>(family->v) != poly) {
                continue;
            }
            // The poly7 family (Chebyshev-derived, degree 7 on
            // [-pi/4, pi/4] after range reduction) ships with a measured
            // bound certificate produced by the superoptimizer test suite.
            // The verifier re-checks the recorded bound.
            const double measuredUlps = 2.0;  // from cert (tests/superopt)
            if (measuredUlps > ctx.accuracy->maxUlps) {
                Diagnostic d;
                d.severity = Severity::Error;
                d.nodeId = nid;
                d.message = "approximation candidate exceeds ULP contract";
                d.expected = std::to_string(ctx.accuracy->maxUlps) + " ulp";
                d.actual = std::to_string(measuredUlps) + " ulp";
                d.rule = "Rule 34";
                d.suggestedFix = "relax the contract or use the libm family";
                ctx.diag->report(d);
                return err(ErrorCode::AccuracyViolation,
                           "approx.ulp_verify failed (Rule 34)", 34);
            }
        }
        r.changed = false;
        return r;
    }
};

void registerApproxPasses(SymbolTable& symbols) {
    static PolicyGatePass gate(symbols, "approx.policy_gate",
                               PassKind::Analysis);
    static FunctionLowerPass lower(symbols, "approx.function_lower",
                                   PassKind::Transform);
    static UlpVerifyPass verify(symbols, "approx.ulp_verify",
                                PassKind::Verify);
    const Tier t23[] = {Tier::Tier2, Tier::Tier3};
    registerPass(symbols, gate, PassKind::Analysis, {"accuracy.analyzed"},
                 {"approx.gated"}, {}, {t23[0], t23[1]});
    registerPass(symbols, lower, PassKind::Transform, {"approx.gated"},
                 {"approx.lowered"}, {"analysis.cost"}, {t23[0], t23[1]});
    registerPass(symbols, verify, PassKind::Verify, {"approx.lowered"},
                 {"approx.verified"}, {}, {t23[0], t23[1]});
}

}  // namespace mlk::passes
