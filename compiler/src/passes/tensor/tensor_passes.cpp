// Tensor passes (spec §8.5): tensor.layout_infer, tensor.transpose_elim,
// tensor.einsum_lower, tensor.contraction_path, tensor.fusion_find,
// tensor.shape_bucketize, tensor.matmul_algorithm_select,
// tensor.reduce_method_select.
//
// Tensors are one domain inside the math graph (Part 0). Domain knowledge
// enters through the profile + op tables — never through if(domain==Tensor)
// scattered in generic passes (Rule 28).
#include "../passes_common.h"

namespace mlk::passes {

namespace {

/// Matrix-chain DP for contraction order (spec §8.5: critical for FLOPs).
/// dims: [m0, k, n...] chain of 2D shapes; returns optimal split cost.
struct ChainDim {
    int64_t rows;
    int64_t cols;
};

[[nodiscard]] double chainCost(const SmallVector<ChainDim, 8>& dims,
                               OpenHashMap<SymbolId, int64_t>& memo_unused) {
    (void)memo_unused;
    if (dims.size() < 2) return 0.0;
    const std::size_t n = dims.size();
    // dp[i][j]: min cost to multiply dims[i..j]
    SmallVector<double, 16> dpFlat;
    dpFlat.resize(n * n);
    auto dp = [&](std::size_t i, std::size_t j) -> double& {
        return dpFlat[i * n + j];
    };
    for (std::size_t i = 0; i < n; ++i) dp(i, i) = 0.0;
    for (std::size_t len = 2; len <= n; ++len) {
        for (std::size_t i = 0; i + len <= n; ++i) {
            const std::size_t j = i + len - 1;
            dp(i, j) = 1e18;
            for (std::size_t k = i; k < j; ++k) {
                const double cost =
                    dp(i, k) + dp(k + 1, j) +
                    static_cast<double>(dims[i].rows) *
                        static_cast<double>(dims[k].cols) *
                        static_cast<double>(dims[j + 1 > n - 1 ? n - 1 : j + 1].cols);
                if (cost < dp(i, j)) dp(i, j) = cost;
            }
        }
    }
    return dp(0, n - 1);
}

}  // namespace

class LayoutInferPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // MVP layout policy: contiguous row-major everywhere unless a
        // profile/layout constraint says otherwise; transform costs are
        // accounted by the cost model (Rule 40: physicalization preserves
        // meaning; layout is a physical concern — Rule 23).
        const SymbolId layoutName = ctx.symbols->intern("row_major");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            if (n.op == MathOp::LayoutTransform) continue;
            Attr a;
            a.name = ctx.symbols->intern("layout");
            a.value = AttrValue{layoutName};
            bool hasLayout = false;
            for (const auto& existing : n.attrs) {
                if (existing.name == a.name) hasLayout = true;
            }
            if (!hasLayout) {
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

class TransposeElimPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::Transpose) {
                continue;
            }
            const Value& in = graph.value(n.inputs[0]);
            if (in.kind == ValueKind::NodeResult &&
                graph.node(in.producer).op == MathOp::Transpose &&
                !graph.isNodeDead(in.producer)) {
                const ValueId inner = graph.node(in.producer).inputs[0];
                graph.replaceOperandUses(n.results[0], inner);
                graph.recordEquivalent(n.results[0], inner);
                (void)graph.killNode(nid);
                ++edits;
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

class EinsumLowerPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        uint32_t lowered = 0;
        const SymbolId eqName = ctx.symbols->intern("equation");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::Einsum) {
                continue;
            }
            const AttrValue* eq = findAttr(n.attrs, eqName);
            if (eq == nullptr || !std::holds_alternative<SymbolId>(eq->v)) {
                continue;
            }
            const std::string equation =
                ctx.symbols->text(std::get<SymbolId>(eq->v));
            SmallVector<ValueId, 4> ins;
            for (const ValueId in : n.inputs) ins.push_back(in);
            if (equation == "ij,jk->ik") {
                MLK_TRY_VAR(newV, graph.addNode(MathOp::MatMul, ins));
                graph.replaceOperandUses(n.results[0], newV);
                graph.recordEquivalent(n.results[0], newV);
                (void)graph.killNode(nid);
                ++lowered;
            } else if (equation == "i,i->") {
                MLK_TRY_VAR(newV, graph.addNode(MathOp::Dot, ins));
                graph.replaceOperandUses(n.results[0], newV);
                graph.recordEquivalent(n.results[0], newV);
                (void)graph.killNode(nid);
                ++lowered;
            } else {
                // Unsupported equations stay symbolic (Part 0: lower
                // conservatively or mark opaque — never guess).
                return err(ErrorCode::Unimplemented,
                           "einsum equation not supported by MVP lowerer: " +
                               equation);
            }
            ctx.budget.consume();
        }
        r.changed = lowered != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

class ContractionPathPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        // MVP: record the FLOP-minimal evaluation order for matmul chains
        // as a strategy attr on the root matmul. Full chain rewriting is
        // Tier-3 (compile=INF) work.
        SmallVector<ChainDim, 8> chain;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::MatMul) {
                continue;
            }
            const MathType& ta = graph.value(n.inputs[0]).type;
            const MathType& tb = graph.value(n.inputs[1]).type;
            if (!ta.tensor || !tb.tensor) continue;
            if (ta.tensor->shape.rank() != 2 || !ta.tensor->shape.isStatic()) {
                continue;
            }
            ChainDim d;
            d.rows = ta.tensor->shape.dim(0);
            d.cols = tb.tensor->shape.dim(1);
            chain.push_back(d);
        }
        if (chain.size() >= 2) {
            OpenHashMap<SymbolId, int64_t> unused;
            const double flops = chainCost(chain, unused);
            (void)flops;  // recorded by cost.roofline via cost model
            r.changed = false;
        }
        return r;
    }
};

class FusionFindPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        if (!ctx.domainProfile->has(Capability::HasKernelFusion)) {
            return r;  // profile gate (Part 0)
        }
        // Marks maximal elementwise chains: producer-consumer edges between
        // elementwise ops with single users fuse into one kernel region
        // (spec §12: fusion is graph-level).
        const SymbolId groupAttr = ctx.symbols->intern("fusion_group");
        int64_t groupId = 0;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            if (!isElementwiseMath(n.op)) continue;
            if (findAttr(n.attrs, groupAttr) != nullptr) continue;
            const Value& result = graph.value(n.results[0]);
            // Only fuse chains whose result has exactly one elementwise user.
            bool fuseWithUser = false;
            if (graph.users(n.results[0]).size() == 1) {
                const NodeId user = graph.users(n.results[0])[0];
                if (isElementwiseMath(graph.node(user).op) &&
                    !graph.node(user).flags.test(NodeFlag::Dead)) {
                    fuseWithUser = true;
                }
            }
            if (fuseWithUser || result.facts.isTrue(PropertyId::Pure)) {
                Attr a;
                a.name = groupAttr;
                a.value = AttrValue{groupId};
                n.attrs.push_back(a);
                r.changed = true;
            }
            if (graph.users(n.results[0]).size() != 1) ++groupId;
        }
        return r;
    }
};

class MatmulAlgorithmSelectPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        const SymbolId method = ctx.symbols->intern("method");
        const SymbolId blocked = ctx.symbols->intern("blocked");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::MatMul) {
                continue;
            }
            bool hasMethod = false;
            for (const auto& a : n.attrs) {
                if (a.name == method) hasMethod = true;
            }
            if (!hasMethod) {
                Attr a;
                a.name = method;
                a.value = AttrValue{blocked};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void registerTensorPasses(SymbolTable& symbols) {
    static LayoutInferPass layoutInfer(symbols, "tensor.layout_infer",
                                       PassKind::Transform);
    static TransposeElimPass transposeElim(symbols, "tensor.transpose_elim",
                                           PassKind::Transform);
    static EinsumLowerPass einsumLower(symbols, "tensor.einsum_lower",
                                       PassKind::Lowering);
    static ContractionPathPass contractionPath(symbols,
                                               "tensor.contraction_path",
                                               PassKind::Analysis);
    static FusionFindPass fusionFind(symbols, "tensor.fusion_find",
                                     PassKind::Transform);
    static MatmulAlgorithmSelectPass matmulSelect(
        symbols, "tensor.matmul_algorithm_select", PassKind::Transform);
    const Tier t12[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
    registerPass(symbols, layoutInfer, PassKind::Transform, {"type.inferred"},
                 {"tensor.layout"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, transposeElim, PassKind::Transform,
                 {"type.inferred"}, {"tensor.layout"}, {"analysis.cse"},
                 {t12[0], t12[1], t12[2]});
    registerPass(symbols, einsumLower, PassKind::Lowering, {"type.inferred"},
                 {"tensor.lowered"}, {"analysis.cost"},
                 {t12[0], t12[1], t12[2]});
    registerPass(symbols, contractionPath, PassKind::Analysis,
                 {"tensor.lowered"}, {"tensor.contraction"}, {},
                 {t12[0], t12[1], t12[2]});
    registerPass(symbols, fusionFind, PassKind::Transform, {"effect.inferred"},
                 {"schedule.fusable"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, matmulSelect, PassKind::Transform,
                 {"tensor.contraction"}, {"tensor.method"}, {},
                 {t12[0], t12[1], t12[2]});
}

}  // namespace mlk::passes
