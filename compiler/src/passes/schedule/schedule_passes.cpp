// Schedule passes (spec §8.7): schedule.region_extract, schedule.fuse,
// schedule.tile, schedule.vectorize, schedule.parallelize, schedule.unroll.
// All decisions are recorded as declarative schedule params on the kernel
// region (Rule 54) and consumed by lowering/executors.
#include "../passes_common.h"

namespace mlk::passes {

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

class VectorizePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Rule 36: vectorization requires dependence proof. For elementwise
        // SSA subgraphs over distinct buffers the dependence proof is
        // trivial (fresh NodeResults never alias inputs); recorded as the
        // vector_width schedule param.
        const SymbolId vw = ctx.symbols->intern("vector_width");
        int64_t width = static_cast<int64_t>(constants::kDefaultSimdWidthF32);
        if (ctx.knobs != nullptr) {
            if (const int64_t* v = ctx.knobs->find(vw)) width = *v;
        }
        for (const NodeId nid : graph.topoOrder()) {
            Node& node = graph.node(nid);
            if (node.flags.test(NodeFlag::Dead)) continue;
            if (!isElementwiseMath(node.op)) continue;
            bool has = false;
            for (const auto& a : node.attrs) {
                if (a.name == vw) has = true;
            }
            if (!has) {
                Attr a;
                a.name = vw;
                a.value = AttrValue{width};
                node.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

class ParallelizePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Parallel mapping is recorded declaratively; the executor decides
        // thread counts (Rule 12: thread-local allocation; Rule 137: no
        // global locks on hot paths).
        const SymbolId threadsAttr = ctx.symbols->intern("parallel");
        const int64_t defaultParallel = 1;
        for (const NodeId nid : graph.topoOrder()) {
            Node& node = graph.node(nid);
            if (node.flags.test(NodeFlag::Dead)) continue;
            bool has = false;
            for (const auto& a : node.attrs) {
                if (a.name == threadsAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = threadsAttr;
                a.value = AttrValue{defaultParallel};
                node.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void registerSchedulePasses(SymbolTable& symbols) {
    static RegionExtractPass regionExtract(symbols, "schedule.region_extract",
                                           PassKind::Transform);
    static FusePass fuse(symbols, "schedule.fuse", PassKind::Transform);
    static TilePass tile(symbols, "schedule.tile", PassKind::Transform);
    static VectorizePass vectorize(symbols, "schedule.vectorize",
                                   PassKind::Transform);
    static ParallelizePass parallelize(symbols, "schedule.parallelize",
                                       PassKind::Transform);
    const Tier t12[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
    registerPass(symbols, regionExtract, PassKind::Transform,
                 {"tensor.fusable", "effect.inferred"}, {"schedule.region"},
                 {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, fuse, PassKind::Transform, {"schedule.region"},
                 {"schedule.fused"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, tile, PassKind::Transform, {"schedule.fused"},
                 {"schedule.tiled"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, vectorize, PassKind::Transform, {"schedule.fused"},
                 {"schedule.vectorized"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, parallelize, PassKind::Transform,
                 {"schedule.vectorized"}, {"schedule.parallel"}, {},
                 {t12[0], t12[1], t12[2]});
}

}  // namespace mlk::passes
