// schedule.tile — declarative tiling params on matmul nodes (spec §8.7;
// Rule 29: empirically validated defaults, overridable via knobs, tuned by
// the autotuner; Rule 54: recorded as declarative schedule params).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kSchedTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class TilePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Heuristic tiling per Rule 29 (empirically validated defaults from
        // constants; overridable via knobs; tuned by the autotuner).
        const SymbolId tileM = ctx.symbols->intern("tile_m");
        const SymbolId tileN = ctx.symbols->intern("tile_n");
        const SymbolId tileK = ctx.symbols->intern("tile_k");
        int64_t m = constants::kDefaultTileM;
        int64_t n = constants::kDefaultTileN;
        int64_t k = constants::kDefaultTileK;
        if (ctx.knobs != nullptr) {
            if (const int64_t* v = ctx.knobs->find(tileM)) m = *v;
            if (const int64_t* v = ctx.knobs->find(tileN)) n = *v;
            if (const int64_t* v = ctx.knobs->find(tileK)) k = *v;
        }
        for (const NodeId nid : graph.topoOrder()) {
            Node& node = graph.node(nid);
            if (node.flags.test(NodeFlag::Dead) || node.op != MathOp::MatMul) {
                continue;
            }
            bool exists = false;
            for (const auto& a : node.attrs) {
                if (a.name == tileM) exists = true;
            }
            if (!exists) {
                Attr ma;
                ma.name = tileM;
                ma.value = AttrValue{m};
                Attr na;
                na.name = tileN;
                na.value = AttrValue{n};
                Attr ka;
                ka.name = tileK;
                ka.value = AttrValue{k};
                node.attrs.push_back(ma);
                node.attrs.push_back(na);
                node.attrs.push_back(ka);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_schedule_tile_pass(SymbolTable& symbols) {
    static TilePass pass(symbols, "schedule.tile", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"schedule.fused"},
                 {"schedule.tiled"}, {},
                 {kSchedTiers[0], kSchedTiers[1], kSchedTiers[2]});
}

}  // namespace mlk::passes
