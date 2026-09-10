// ir.verify — graph verification pass (spec §8.1, Rule 47).
#include "../passes_common.h"
#include "mlk/verifier/graph_verifier.h"

namespace mlk::passes {

namespace {
constexpr Tier kVerifyTiers[] = {Tier::Tier0, Tier::Tier1, Tier::Tier2,
                                 Tier::Tier3};
}  // namespace

class VerifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = r.nodesAfter = graph.liveNodeCount();
        DiagnosticEngine diag;
        VerifyOptions opts;
        if (!verifyGraph(graph, ctx.domainProfile, opts, diag)) {
            for (const auto& d : diag.entries()) {
                ctx.diag->report(d);
            }
            return err(ErrorCode::VerificationFailed,
                       "ir.verify failed: " +
                           std::to_string(diag.entries().size()) +
                           " diagnostics", 47);
        }
        return r;
    }
};

void register_ir_verify_pass(SymbolTable& symbols) {
    static VerifyPass pass(symbols, "ir.verify", PassKind::Verify);
    registerPass(symbols, pass, PassKind::Verify, {}, {"ir.verified"}, {},
                 {kVerifyTiers[0], kVerifyTiers[1], kVerifyTiers[2],
                  kVerifyTiers[3]});
}

}  // namespace mlk::passes
