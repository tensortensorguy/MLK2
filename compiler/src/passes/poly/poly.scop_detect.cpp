// poly.scop_detect — extract a SCoP (Static Control Part) from the kernel
// module produced by lower.to_kernel_ir / poly.synth (spec §8.13; see
// docs/polyhedral_spec.md). Analysis pass: never mutates the graph or the
// kernel; records the extraction into the polyhedral workspace.
//
// Contract (Rule 142):
//   required  : kernel.built
//   produced  : poly.scop (PolyWorkspace::scop when valid)
//   invalidated: poly.scop (any prior extraction)
//   kill switch: "poly.scop_detect" (Rule 60)
// Fallback (Rules 62/102): an ineligible kernel leaves the workspace
// invalid; downstream poly.* passes become no-ops and the baseline kernel
// is untouched.
#include "../passes_common.h"
#include "mlk/poly/scop.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolyScopDetectPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        if (ctx.killed(nameId(*ctx.symbols))) {
            return r;  // Rule 60: kill switch honored, pipeline continues
        }
        if (ctx.kernelOut == nullptr || ctx.polyWorkspace == nullptr) {
            // Polyhedral pipeline not engaged (Tier 1 or embedding tool
            // opted out): silent no-op is the documented contract here —
            // the baseline path is authoritative (Rule 62).
            return r;
        }
        poly::PolyWorkspace& ws = *ctx.polyWorkspace;
        ws.scopValid = false;
        ws.dependencesValid = false;
        ws.scheduleValid = false;
        ws.tileValid = false;
        ws.codegenValid = false;

        MLK_TRY_VAR(scop, poly::extractScop(*ctx.kernelOut, *ctx.symbols));
        ws.scop = std::move(scop);
        ws.scopValid = true;
        ws.statsStatements =
            static_cast<uint32_t>(ws.scop.statements.size());
        return r;
    }
};

void register_poly_scop_detect_pass(SymbolTable& symbols) {
    static PolyScopDetectPass pass(symbols, "poly.scop_detect",
                                   PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"kernel.built"},
                 {"poly.scop"}, {"poly.scop"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
