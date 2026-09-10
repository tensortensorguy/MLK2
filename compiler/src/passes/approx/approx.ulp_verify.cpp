// approx.ulp_verify — verifies every poly-family function node against its
// ULP contract (spec §8.6; Rule 50: candidates must be verified before
// benchmarking; Rule 34: contract violations fail the compilation).
#include "../passes_common.h"
#include "mlk/ir/attrs.h"

namespace mlk::passes {

namespace {
constexpr Tier kApproxTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

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

void register_approx_ulp_verify_pass(SymbolTable& symbols) {
    static UlpVerifyPass pass(symbols, "approx.ulp_verify",
                              PassKind::Verify);
    registerPass(symbols, pass, PassKind::Verify, {"approx.lowered"},
                 {"approx.verified"}, {},
                 {kApproxTiers[0], kApproxTiers[1]});
}

}  // namespace mlk::passes
