// math.canonicalize — composition driver (spec §8.2):
//   normalize_ops -> commutative_sort -> constant_fold -> identity_elim,
// iterated to fixpoint under budget (Rule 10).
//
// The driver resolves its sub-passes through the registry at run time, so
// this file has no compile-time coupling to the sub-pass translation units
// (1:1 pass->file; sub-passes remain individually registered and killable,
// Rule 60 — each kill switch is honored below).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};

/// Fixed sub-pipeline, in spec §8.2 order.
constexpr const char* kSubPasses[] = {
    "math.normalize_ops", "math.commutative_sort", "math.constant_fold",
    "math.identity_elim",
};
}  // namespace

class CanonicalizePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        bool changed = false;
        for (uint32_t iter = 0; iter < ctx.budget.fixpointIterations; ++iter) {
            bool any = false;
            for (const char* subName : kSubPasses) {
                const SymbolId sid = ctx.symbols->intern(subName);
                if (ctx.killed(sid)) continue;
                Pass* sub = PassRegistry::instance().byName(*(ctx.symbols), sid);
                if (sub == nullptr) {
                    return err(ErrorCode::Internal,
                               std::string(subName) + " not registered");
                }
                MLK_TRY_VAR(subResult, sub->run(ctx, graph));
                any = any || subResult.changed;
            }
            changed = changed || any;
            if (!any) break;
        }
        r.changed = changed;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_canonicalize_pass(SymbolTable& symbols) {
    static CanonicalizePass pass(symbols, "math.canonicalize",
                                 PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.canonical"}, {"analysis.cse"},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
