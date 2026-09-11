// poly.verify — verify the transformed kernel before it reaches the
// backend (spec §8.13; docs/polyhedral_spec.md §verification). Verify
// pass: re-runs the schedule legality proof against the dependence
// relations and restores the baseline kernel on any failure (Rules
// 62/102/115: fallback, never a broken realization).
//
// Contract (Rule 142):
//   required  : poly.codegen
//   produced  : poly.kernel.verified
//   invalidated: poly.codegen (on restore)
//   kill switch: "poly.verify" (Rule 60) — skipping verification is a
//   documented opt-out for debugging; production pipelines keep it on.
#include "../passes_common.h"
#include "mlk/poly/pluto.h"
#include "mlk/poly/workspace.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolyVerifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        if (ctx.killed(nameId(*ctx.symbols))) return r;  // Rule 60
        if (ctx.kernelOut == nullptr || ctx.polyWorkspace == nullptr) {
            return r;
        }
        poly::PolyWorkspace& ws = *ctx.polyWorkspace;
        if (!ws.codegenValid || !ws.baselineSaved) return r;

        // 1. Static legality: every dependence strictly ordered by the
        // schedule (the emitted loops execute exactly that order).
        MLK_TRY_VAR(legal,
                    poly::verifyScheduleLegality(ws.scop, ws.dependences,
                                                 ws.schedule));
        if (!legal) {
            *ctx.kernelOut = ws.baselineKernel;
            ws.codegenValid = false;
            ws.scheduleValid = false;
            ws.tileValid = false;
            return r;  // baseline restored; pipeline continues
        }

        // 2. Structural sanity: the transformed forest references valid
        // buffer ids and every statement got emitted (payload count ==
        // statement count via Compute nodes carrying the chains).
        uint32_t computeCount = 0;
        for (const KernelNode& n : ctx.kernelOut->nodes) {
            if (n.op == KernelOp::Compute) ++computeCount;
            if (n.bufferOut != constants::kInvalidId &&
                n.bufferOut >= ctx.kernelOut->buffers.size()) {
                *ctx.kernelOut = ws.baselineKernel;
                ws.codegenValid = false;
                return err(ErrorCode::VerificationFailed,
                           "poly.verify: store references an invalid "
                           "buffer id",
                           47);
            }
        }
        if (computeCount != ws.scop.statements.size()) {
            *ctx.kernelOut = ws.baselineKernel;
            ws.codegenValid = false;
            return err(ErrorCode::VerificationFailed,
                       "poly.verify: emitted compute count does not match "
                       "the statement count",
                       47);
        }
        return r;
    }
};

void register_poly_verify_pass(SymbolTable& symbols) {
    static PolyVerifyPass pass(symbols, "poly.verify", PassKind::Verify);
    registerPass(symbols, pass, PassKind::Verify, {"poly.codegen"},
                 {"poly.kernel.verified"}, {"poly.codegen"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
