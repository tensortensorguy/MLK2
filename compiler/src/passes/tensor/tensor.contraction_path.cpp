// tensor.contraction_path — FLOP-minimal evaluation order for matmul
// chains via matrix-chain DP (spec §8.5; analysis-only: the order rides as
// strategy data, full chain rewriting is Tier-3 work — and Rule 33-gated,
// since reassociating an FP matmul chain changes rounding).
//
// Round-23 repair (ADR-0009): the previous implementation (a) modeled the
// chain on the WRONG dims — it collected every MatMul's (rows, cols) pair
// independently and ran a DP over cols that never touched the contraction
// dims, so the "optimal cost" was meaningless; (b) discarded the result
// into a local. The DP is now the textbook matrix-chain recurrence over
// the shared dimension vector p[] of ACTUAL chains (matmul feeding a
// matmul), and the result is attached to the IR: each chain node carries
// its subtree's optimal split ("contraction_split") and cost
// ("contraction_flops") as strategy attrs. No lowering consumes the order
// yet — realization requires a reassociation-legality contract (Rule 33);
// the recorded decision and gap vs the naive left-deep order are in
// docs/passes/tensor.contraction_path.md.
#include "../passes_common.h"
#include "mlk/core/event_sink.h"

#include <cmath>

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};

/// A maximal matmul chain m1 = M1*M2, m2 = m1*M3, ...: node ids of the
/// matmuls in chain order (m1 first, root last) plus the shared dimension
/// vector p[0..n] (p[0] = M1 rows, p[i] = M_{i} cols = M_{i+1} rows).
struct MatmulChain {
    SmallVector<NodeId, 8> nodes;  // m1..mn (left-deep)
    SmallVector<int64_t, 9> p;     // n+1 shared dims
};

[[nodiscard]] bool staticDims(const MathGraph& g, ValueId v, int64_t& rows,
                              int64_t& cols) {
    const MathType& t = g.value(v).type;
    if (!t.tensor || t.tensor->shape.rank() != 2 ||
        !t.tensor->shape.isStatic()) {
        return false;
    }
    rows = t.tensor->shape.dim(0);
    cols = t.tensor->shape.dim(1);
    return true;
}

/// Live-user count (fusion_find discipline: users_ of dead nodes are
/// reclaimed lazily, so filter by the Dead flag).
[[nodiscard]] std::size_t liveUserCount(const MathGraph& g, ValueId v) {
    std::size_t n = 0;
    for (const NodeId u : g.users(v)) {
        if (!g.node(u).flags.test(NodeFlag::Dead)) ++n;
    }
    return n;
}

/// Walks up from a matmul node through its LEFT matmul inputs, collecting
/// the maximal left-deep chain ending at `root`. Returns the chain in
/// m1..mn order with the shared-dim vector, or false when the dims are
/// inconsistent (dynamic/incompatible — no analysis is recorded).
[[nodiscard]] bool collectChain(const MathGraph& g, NodeId root,
                                MatmulChain& out) {
    SmallVector<NodeId, 8> rev;  // root-first
    NodeId cur = root;
    while (true) {
        const Node& n = g.node(cur);
        if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::MatMul) {
            return false;
        }
        rev.push_back(cur);
        const Value& lhs = g.value(n.inputs[0]);
        if (lhs.kind != ValueKind::NodeResult) break;
        const Node& producer = g.node(lhs.producer);
        if (producer.op != MathOp::MatMul || g.isNodeDead(lhs.producer)) {
            break;
        }
        // Chain membership requires the producer to have exactly this user:
        // a shared intermediate is a materialized value, not a chain link.
        if (liveUserCount(g, lhs.id) != 1) break;
        cur = lhs.producer;
    }
    // Build p[] from the leaf matrices: M1 = leftmost operand of rev.back().
    const std::size_t n = rev.size();
    SmallVector<int64_t, 9> p;
    {
        // Left operand of the FIRST chain node (m1).
        const Node& first = g.node(rev[n - 1]);
        int64_t r = 0;
        int64_t c = 0;
        if (!staticDims(g, first.inputs[0], r, c)) return false;
        p.push_back(r);
        p.push_back(c);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const Node& m = g.node(rev[n - 1 - i]);  // m1, m2, ..., mn
        int64_t r = 0;
        int64_t c = 0;
        if (!staticDims(g, m.inputs[1], r, c)) return false;
        if (r != p.back()) return false;  // inconsistent chain dims
        if (i + 1 < n) {
            // The NEXT matmul consumes this node's result: its shape must
            // be [left.rows, right.cols] = [p[p.size()-2], c] — i.e. the
            // shared dim r cancels and does NOT reappear in the result.
            int64_t rr = 0;
            int64_t rc = 0;
            if (!staticDims(g, m.results[0], rr, rc)) return false;
            if (rr != p[p.size() - 2] || rc != c) return false;
        }
        p.push_back(c);
    }
    out.nodes.clear();
    for (std::size_t i = 0; i < n; ++i) out.nodes.push_back(rev[n - 1 - i]);
    out.p = std::move(p);
    return true;
}

/// Textbook matrix-chain DP over p[] (matrices 1..n, p has n+1 entries).
/// memo/split are flat (n+1)x(n+1) tables (SmallVector has no assign; flat
/// indexing keeps a single allocation). memo[i*T+j]: optimal cost for
/// matrices i..j; split[i*T+j]: the top-level split (0 when j == i).
void chainDp(const SmallVector<int64_t, 9>& p, std::size_t n,
             SmallVector<double, 64>& memo,
             SmallVector<std::size_t, 64>& split) {
    const std::size_t T = n + 1;
    memo.resize(T * T);
    split.resize(T * T);
    for (std::size_t i = 0; i < T; ++i) {
        for (std::size_t j = 0; j < T; ++j) {
            memo[i * T + j] = 0.0;
            split[i * T + j] = 0;
        }
    }
    for (std::size_t len = 2; len <= n; ++len) {
        for (std::size_t i = 1; i + len - 1 <= n; ++i) {
            const std::size_t j = i + len - 1;
            memo[i * T + j] = 1e18;
            for (std::size_t k = i; k < j; ++k) {
                const double cost =
                    memo[i * T + k] + memo[(k + 1) * T + j] +
                    static_cast<double>(p[i - 1]) *
                        static_cast<double>(p[k]) *
                        static_cast<double>(p[j]);
                if (cost < memo[i * T + j]) {
                    memo[i * T + j] = cost;
                    split[i * T + j] = k;
                }
            }
        }
    }
}
}  // namespace

class ContractionPathPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        const SymbolId splitAttr = ctx.symbols->intern("contraction_split");
        const SymbolId flopsAttr = ctx.symbols->intern("contraction_flops");
        const SymbolId naiveAttr =
            ctx.symbols->intern("contraction_flops_naive");
        uint32_t chains = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::MatMul) {
                continue;
            }
            // Chain ROOTS only: a matmul result consumed as the LEFT input
            // of another matmul is an interior node of a longer chain and
            // is handled from that root. (Right-operand consumption is not
            // a left-deep chain link; the node stays its own root.)
            const Value& res = graph.value(n.results[0]);
            if (res.kind == ValueKind::NodeResult &&
                graph.node(res.producer).op == MathOp::MatMul &&
                !graph.isNodeDead(res.producer) &&
                graph.node(res.producer).inputs[0] == res.id &&
                liveUserCount(graph, res.id) == 1) {
                continue;
            }
            if (findAttr(n.attrs, splitAttr) != nullptr) continue;  // idempotent
            MatmulChain chain;
            if (!collectChain(graph, nid, chain) || chain.nodes.size() < 2) {
                continue;
            }
            const std::size_t nodes = chain.nodes.size();
            // The chain has `nodes` matmul nodes but `nodes + 1` LEAF
            // matrices (node j = product of matrices 1..j+1); the DP runs
            // over the matrices.
            const std::size_t matrices = nodes + 1;
            const std::size_t T = matrices + 1;
            SmallVector<double, 64> memo;
            SmallVector<std::size_t, 64> split;
            chainDp(chain.p, matrices, memo, split);
            // Naive left-deep cost for the recorded gap.
            double naive = 0.0;
            for (std::size_t k = 2; k <= matrices; ++k) {
                naive += static_cast<double>(chain.p[0]) *
                         static_cast<double>(chain.p[k - 1]) *
                         static_cast<double>(chain.p[k]);
            }
            // Attach per-node subtree data: node j covers matrices 1..j+1
            // (m1 = M1*M2, m2 = m1*M3, ...).
            for (std::size_t j = 1; j <= nodes; ++j) {
                Node& m = graph.node(chain.nodes[j - 1]);
                if (findAttr(m.attrs, splitAttr) == nullptr) {
                    Attr a;
                    a.name = splitAttr;
                    a.value =
                        AttrValue{static_cast<int64_t>(split[T + j + 1])};
                    m.attrs.push_back(a);
                }
                if (findAttr(m.attrs, flopsAttr) == nullptr) {
                    Attr a;
                    a.name = flopsAttr;
                    a.value = AttrValue{memo[T + j + 1]};
                    m.attrs.push_back(a);
                }
            }
            Node& rootNode = graph.node(chain.nodes[nodes - 1]);
            if (findAttr(rootNode.attrs, naiveAttr) == nullptr) {
                Attr a;
                a.name = naiveAttr;
                a.value = AttrValue{naive};
                rootNode.attrs.push_back(a);
            }
            ++chains;
        }
        if (ctx.telemetry != nullptr && chains > 0) {
            ctx.telemetry->event(
                11, nameId(*ctx.symbols),
                ctx.symbols->intern("contraction_chains"), chains);
        }
        r.changed = false;  // analysis: strategy attrs only, no rewiring
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_tensor_contraction_path_pass(SymbolTable& symbols) {
    static ContractionPathPass pass(symbols, "tensor.contraction_path",
                                    PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"tensor.lowered"},
                 {"tensor.contraction"}, {},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
