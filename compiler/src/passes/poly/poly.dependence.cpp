// poly.dependence — exact data dependence analysis over the extracted SCoP
// (spec §8.13; docs/polyhedral_spec.md §dependences). Analysis pass: fills
// PolyWorkspace::dependences with live RAW/WAR/WAW relations.
//
// Contract (Rule 142):
//   required  : poly.scop
//   produced  : poly.deps
//   invalidated: poly.deps, poly.schedule, poly.tile, poly.codegen
//   kill switch: "poly.dependence" (Rule 60)
#include "../passes_common.h"
#include "mlk/poly/workspace.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolyDependencePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        if (ctx.killed(nameId(*ctx.symbols))) return r;  // Rule 60
        if (ctx.kernelOut == nullptr || ctx.polyWorkspace == nullptr) {
            return r;  // poly pipeline not engaged (documented no-op)
        }
        poly::PolyWorkspace& ws = *ctx.polyWorkspace;
        if (!ws.scopValid) return r;  // Rules 62/102: baseline fallback
        ws.dependencesValid = false;
        ws.scheduleValid = false;
        ws.tileValid = false;
        ws.codegenValid = false;

        MLK_TRY_VAR(deps, poly::computeDependences(ws.scop));
        ws.dependences = std::move(deps);
        ws.dependencesValid = true;
        ws.statsDependences =
            static_cast<uint32_t>(ws.dependences.size());
        return r;
    }
};

void register_poly_dependence_pass(SymbolTable& symbols) {
    static PolyDependencePass pass(symbols, "poly.dependence",
                                   PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"poly.scop"},
                 {"poly.deps"},
                 {"poly.deps", "poly.schedule", "poly.tile", "poly.codegen"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
