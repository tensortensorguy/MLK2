// poly.codegen — regenerate the KernelModule node tree from the schedule
// (spec §8.13; docs/polyhedral_spec.md §codegen). Lowering pass: replaces
// ctx.kernelOut's nodes with the transformed loop forest; buffers and
// schedule params are preserved.
//
// Contract (Rule 142):
//   required  : poly.scop, poly.deps, poly.schedule, poly.tile
//   produced  : poly.kernel.transformed
//   invalidated: poly.codegen
//   kill switch: "poly.codegen" (Rule 60)
// Fallback (Rules 62/102): codegen bail (non-unimodular rows, fission
// across dims, undivisible tile sizes) leaves the baseline kernel intact
// and invalidates the poly pipeline results.
#include "../passes_common.h"
#include "mlk/poly/codegen.h"
#include "mlk/poly/workspace.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolyCodegenPass final : public PassBase {
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
        if (!ws.scopValid || !ws.dependencesValid || !ws.scheduleValid ||
            !ws.tileValid) {
            return r;
        }
        ws.codegenValid = false;

        MLK_TRY_VAR(kernel,
                    poly::emitScheduledKernel(ws.scop, ws.schedule,
                                              ws.tiled, *ctx.kernelOut,
                                              *ctx.symbols));
        *ctx.kernelOut = std::move(kernel);
        ws.codegenValid = true;
        r.changed = true;
        return r;
    }
};

void register_poly_codegen_pass(SymbolTable& symbols) {
    static PolyCodegenPass pass(symbols, "poly.codegen", PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering,
                 {"poly.scop", "poly.deps", "poly.schedule", "poly.tile"},
                 {"poly.kernel.transformed"}, {"poly.codegen"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
