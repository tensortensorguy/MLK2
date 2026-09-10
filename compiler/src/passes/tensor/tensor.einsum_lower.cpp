// tensor.einsum_lower — einsum equation -> primitive ops (spec §8.5:
// ij,jk->ik => matmul, i,i-> => dot; unsupported equations stay symbolic
// or error — never guessed, Part 0/Rule 28).
#include "../passes_common.h"
#include "mlk/ir/attrs.h"

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

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

void register_tensor_einsum_lower_pass(SymbolTable& symbols) {
    static EinsumLowerPass pass(symbols, "tensor.einsum_lower",
                                PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering, {"type.inferred"},
                 {"tensor.lowered"}, {"analysis.cost"},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
