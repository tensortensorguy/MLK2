// schedule.fuse — fusion legality/decision stage (spec §8.7; realized at
// lowering: elementwise chains sharing a fusion_group collapse into one
// loop body, see lower.to_kernel_ir).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kSchedTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class FusePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        if (!ctx.domainProfile->has(Capability::HasKernelFusion)) return r;
        // Fusion is realized at lowering: elementwise chains sharing a
        // fusion_group collapse into one loop body (see lower.to_kernel_ir).
        r.changed = false;
        return r;
    }
};

void register_schedule_fuse_pass(SymbolTable& symbols) {
    static FusePass pass(symbols, "schedule.fuse", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"schedule.region"},
                 {"schedule.fused"}, {},
                 {kSchedTiers[0], kSchedTiers[1], kSchedTiers[2]});
}

}  // namespace mlk::passes
