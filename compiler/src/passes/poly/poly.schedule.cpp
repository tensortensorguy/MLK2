// poly.schedule — Pluto-style affine schedule synthesis over the SCoP and
// its dependence set (spec §8.13; docs/polyhedral_spec.md §scheduling).
// Transform pass: writes PolySchedule into the workspace; the kernel is
// rewritten later by poly.codegen.
//
// Contract (Rule 142):
//   required  : poly.scop, poly.deps
//   produced  : poly.schedule
//   invalidated: poly.schedule, poly.tile, poly.codegen
//   kill switch: "poly.schedule" (Rule 60)
// Markings: outermost parallel rows feed schedule.parallelize; the
// innermost parallel row feeds schedule.vectorize (Rule 36 gates live in
// the consumer passes / executor).
#include "../passes_common.h"
#include "mlk/poly/pluto.h"
#include "mlk/poly/workspace.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolySchedulePass final : public PassBase {
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
        if (!ws.scopValid || !ws.dependencesValid) return r;
        ws.scheduleValid = false;
        ws.tileValid = false;
        ws.codegenValid = false;

        // "No schedule found" (exact-LP infeasibility, realizability
        // gate rejections, budgets) is a FALLBACK, not a pipeline
        // failure (Rules 62/102/115): the kernel keeps its current —
        // semantically correct — form (for synthesized classes that is
        // the materialized band nest, e.g. softmax) and downstream poly
        // passes no-op on the invalid schedule. The scheduler's own
        // internal contract violations still propagate as errors (they
        // are engine bugs, not capability boundaries).
        auto sched = poly::computePlutoSchedule(ws.scop, ws.dependences);
        if (!sched.has_value()) {
            if (ctx.diag != nullptr) {
                Diagnostic d;
                d.severity = Severity::Note;
                d.pass = nameId(*ctx.symbols);
                d.message =
                    "poly.schedule: no legal schedule found — kernel "
                    "keeps its current (correct) form: " +
                    sched.error().message;
                ctx.diag->report(std::move(d));
            }
            return r;
        }
        ws.statsScheduleRows = static_cast<uint32_t>(sched->rows.size());
        ws.schedule = std::move(*sched);
        ws.scheduleValid = true;
        return r;
    }
};

void register_poly_schedule_pass(SymbolTable& symbols) {
    static PolySchedulePass pass(symbols, "poly.schedule",
                                 PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform,
                 {"poly.scop", "poly.deps"}, {"poly.schedule"},
                 {"poly.schedule", "poly.tile", "poly.codegen"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
