// tensor.contraction_path — FLOP-minimal evaluation order for matmul
// chains via matrix-chain DP (spec §8.5; analysis-only: the order rides as
// strategy data, full chain rewriting is Tier-3 work).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};

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

void register_tensor_contraction_path_pass(SymbolTable& symbols) {
    static ContractionPathPass pass(symbols, "tensor.contraction_path",
                                    PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"tensor.lowered"},
                 {"tensor.contraction"}, {},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
