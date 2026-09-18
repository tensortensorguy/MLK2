// egraph.extract — saturate then extract the cheapest equivalent form
// (spec §8.3; Rule 21: the original expression stays in the graph and the
// output repointing is the versioned, documented lowering decision).
#include "../passes_common.h"
#include "egraph.h"

namespace mlk::passes {

namespace {
constexpr Tier kEgraphTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class EgraphExtractPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        EGraphConfig cfg;
        cfg.maxNodes = constants::kEgraphDefaultMaxNodes;
        cfg.maxIterations = ctx.budget.fixpointIterations;
        EGraph eg(*ctx.domainProfile, cfg);
        MLK_TRY_VAR(rootClass, eg.importGraph(graph, graph.outputs()[0]));
        for (uint32_t iter = 0; iter < ctx.budget.fixpointIterations; ++iter) {
            MLK_TRY_VAR(grew, eg.saturateOnce(*ctx.symbols));
            if (!grew) break;
        }
        // Extract the cheapest equivalent form and append it to the graph;
        // the output repoints to the extracted root. The original expression
        // stays in the graph (Rule 21); repointing is the versioned,
        // documented lowering decision (recordEquivalent + output swap).
        MathGraph extracted{ctx.symbols};
        MLK_TRY_VAR(best, eg.extract(*ctx.symbols, extracted, rootClass));

        // Transplant with a FULL per-value id map. Extraction materializes
        // in DFS order, so leaves and node results INTERLEAVE (e.g.
        // add(mul(x,2),y) allocates x, 2, mul-result, y, add-result); the
        // previous "leaves first, then results" offset scheme miswired or
        // under-indexed every graph where a leaf follows a node result.
        // Node inputs always precede their node (children materialize
        // first), so two passes over the extracted graph suffice.
        //
        // Named leaves (placeholder/variable/symbol) REUSE the existing
        // graph value of the same kind+name+type instead of creating a
        // duplicate atom: input binding is by value-id order
        // (interpretGraph / kernel ABI), and a re-created "x" would be a
        // second placeholder that no input scalar ever reaches — the tier-3
        // pipeline used to evaluate sin(x^2+3x) at x=2 as sin(0) = 0
        // through exactly that duplication.
        auto reuseNamedLeaf = [&](ValueKind kind, SymbolId name,
                                  const MathType& type) -> ValueId {
            for (const auto& v : graph.values()) {
                if (v.kind == kind && v.name == name && v.type == type) {
                    return v.id;
                }
            }
            return kInvalidValueId;
        };
        SmallVector<ValueId, 16> mapped(extracted.numValues(),
                                        kInvalidValueId);
        for (uint32_t i = 0; i < extracted.numValues(); ++i) {
            const auto& v = extracted.values()[i];
            switch (v.kind) {  // Rule 78: exhaustive
                case ValueKind::Constant:
                    mapped[i] = v.constant.isInt
                                    ? graph.addIntConstant(v.constant.i64,
                                                           v.type)
                                    : graph.addConstant(v.constant.f64,
                                                        v.type);
                    break;
                case ValueKind::Variable:
                case ValueKind::Placeholder:
                case ValueKind::Symbol: {
                    const ValueId existing =
                        reuseNamedLeaf(v.kind, v.name, v.type);
                    mapped[i] = existing != kInvalidValueId
                                    ? existing
                                    : (v.kind == ValueKind::Variable
                                           ? graph.addVariable(v.name, v.type)
                                       : v.kind == ValueKind::Placeholder
                                           ? graph.addPlaceholder(v.name,
                                                                  v.type)
                                           : graph.addSymbol(v.name, v.type));
                    break;
                }
                case ValueKind::NodeResult:
                    break;  // recreated via addNode below
            }
        }
        ValueId extractedRootNew = kInvalidValueId;
        for (const auto& n : extracted.nodes()) {
            SmallVector<ValueId, 4> ins;
            for (const ValueId in : n.inputs) {
                if (in >= extracted.numValues() ||
                    mapped[in] == kInvalidValueId) {
                    return err(ErrorCode::Internal,
                               "extraction produced a dangling input");
                }
                ins.push_back(mapped[in]);
            }
            MLK_TRY_VAR(newV, graph.addNode(n.op, ins, n.attrs));
            mapped[n.results[0]] = newV;
            if (n.results[0] == best) extractedRootNew = newV;
        }
        if (extractedRootNew == kInvalidValueId) {
            return err(ErrorCode::Internal,
                       "extraction lost the root expression");
        }
        // Repoint output: old output value stays alive and recoverable
        // (Rule 21); the outputs list now exposes the extracted form.
        const ValueId oldOutput = graph.outputs()[0];
        graph.recordEquivalent(oldOutput, extractedRootNew);
        graph.outputsRef()[0] = extractedRootNew;
        r.changed = true;
        r.invalidatedAnalyses.push_back(ctx.symbols->intern("analysis.cse"));
        r.invalidatedAnalyses.push_back(ctx.symbols->intern("analysis.cost"));
        return r;
    }
};

void register_egraph_extract_pass(SymbolTable& symbols) {
    static EgraphExtractPass pass(symbols, "egraph.extract",
                                  PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"egraph.saturated"},
                 {"math.extracted"}, {"analysis.cse", "analysis.cost"},
                 {kEgraphTiers[0], kEgraphTiers[1]});
}

}  // namespace mlk::passes
