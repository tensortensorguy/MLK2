// tensor.matmul_algorithm_select — default GEMM method attachment
// (spec §8.5; the choice is a strategy attr consumed by lowering and
// overridable by the autotuner, spec §5.2).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

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

void register_tensor_matmul_algorithm_select_pass(SymbolTable& symbols) {
    static MatmulAlgorithmSelectPass pass(
        symbols, "tensor.matmul_algorithm_select", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform,
                 {"tensor.contraction"}, {"tensor.method"}, {},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
