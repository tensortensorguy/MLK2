// math.cse — hash-based GVN with true structural confirmation (spec §8.2;
// Rule 17: open-addressing map; Rule 79: no deep string dumps; Rule 21:
// merging only for provably identical mathematics).
//
// P0 regression guard: the hash is only a pre-filter. Hash equality must be
// confirmed by structuralEqualValues (attributes, constant payloads, atom
// names, recursive operand structure) — two distinct values must never
// merge (mul(x,x) vs mul(3,x); see structuralValueHash in mlk/ir/value.h).
#include "../passes_common.h"
#include "math_rewrite_utils.h"
#include "mlk/core/constants.h"
#include "mlk/core/hash_map.h"
#include "mlk/ir/graph_hash.h"

#include <utility>

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class CsePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        OpenHashMap<HashValue, ValueId> seen;
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            HashValue h = hashU64(static_cast<uint64_t>(n.op));
            for (const auto& a : n.attrs) {
                h = hashCombine(h, hashU64(a.name));
                h = hashCombine(h, a.value.hash());
            }
            for (const ValueId in : n.inputs) {
                h = hashCombine(h, valueHash(graph, in));
            }
            bool inserted = false;
            ValueId* existing = seen.findOrInsert(h, &inserted, kInvalidValueId);
            if (!inserted && *existing != kInvalidValueId) {
                // Confirm structural equality. Hash equality is only a
                // pre-filter; the confirmation below is a true structural
                // comparison (attributes, constant payloads, atom names,
                // recursive operand structure). Two distinct values must
                // never merge (P0 regression: mul(x,x) vs mul(3,x)).
                if (structurallyEqualValues(graph, *existing, n.results[0], 0)) {
                    MLK_TRYV(replaceResult(graph, nid, *existing));
                    ++edits;
                    ctx.budget.consume();
                    if (ctx.budget.exhausted()) {
                        return err(ErrorCode::BudgetExceeded,
                                   "cse budget exhausted", 131);
                    }
                }
            } else {
                *existing = n.results[0];
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        r.invalidatedAnalyses.push_back(ctx.symbols->intern("analysis.cost"));
        return r;
    }

private:
    // True structural equality between two values, compared recursively
    // through producer nodes with a bounded depth (defensive against
    // accidental cycles from buggy passes; constants compare by payload,
    // named atoms by symbol id, nodes by op + attrs + operand structure).
    // Rule 79: no deep string dumps; Rule 21: merging is only legal for
    // provably identical mathematics.
    [[nodiscard]] static bool structurallyEqualValues(const MathGraph& g,
                                                      ValueId a, ValueId b,
                                                      uint32_t depth) {
        if (a == b) return true;
        if (depth >= constants::kMaxEquivalenceDepth) return false;
        const Value& va = g.value(a);
        const Value& vb = g.value(b);
        if (va.kind != vb.kind) return false;
        if (va.type != vb.type) return false;
        switch (va.kind) {  // Rule 78: exhaustive
            case ValueKind::Constant:
                return va.constant == vb.constant;
            case ValueKind::Variable:
            case ValueKind::Placeholder:
            case ValueKind::Symbol:
                return va.name == vb.name;
            case ValueKind::NodeResult: {
                if (va.producer == kInvalidNodeId ||
                    vb.producer == kInvalidNodeId) {
                    return false;
                }
                const Node& na = g.node(va.producer);
                const Node& nb = g.node(vb.producer);
                if (na.op != nb.op || na.numInputs() != nb.numInputs()) {
                    return false;
                }
                if (na.attrs.size() != nb.attrs.size()) return false;
                for (std::size_t i = 0; i < na.attrs.size(); ++i) {
                    if (na.attrs[i].name != nb.attrs[i].name) return false;
                    if (!(na.attrs[i].value == nb.attrs[i].value)) return false;
                }
                for (std::size_t i = 0; i < na.inputs.size(); ++i) {
                    if (!structurallyEqualValues(g, na.inputs[i], nb.inputs[i],
                                                 depth + 1)) {
                        return false;
                    }
                }
                return true;
            }
        }
        [[assume(false)]];
        std::unreachable();  // exhaustive over ValueKind (Rule 78)
    }
};

void register_math_cse_pass(SymbolTable& symbols) {
    static CsePass pass(symbols, "math.cse", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"math.canonical"},
                 {"analysis.cse"}, {"analysis.cost"},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
