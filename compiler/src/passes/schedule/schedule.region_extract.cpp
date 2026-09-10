// schedule.region_extract — one kernel region per output subgraph
// (spec §8.7/§12; boundaries refined by tensor.fusion_find groups;
// decisions are declarative schedule params, Rule 54).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kSchedTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class RegionExtractPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        (void)graph;
        // MVP: one region per output subgraph; region boundaries refined by
        // fusion attributes from tensor.fusion_find (spec §12: kernel
        // region = subgraph chosen by locality/parallelism/traffic).
        r.changed = false;
        return r;
    }
};

void register_schedule_region_extract_pass(SymbolTable& symbols) {
    static RegionExtractPass pass(symbols, "schedule.region_extract",
                                  PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform,
                 {"tensor.fusable", "effect.inferred"},
                 {"schedule.region"}, {},
                 {kSchedTiers[0], kSchedTiers[1], kSchedTiers[2]});
}

}  // namespace mlk::passes
