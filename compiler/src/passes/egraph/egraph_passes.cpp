// E-graph passes (spec §8.3): egraph.build, egraph.saturate, egraph.extract.
#include "../passes_common.h"
#include "../egraph/egraph.h"
#include "mlk/core/cancellation.h"

namespace mlk::passes {

namespace {
constexpr Tier kEgraphTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class EgraphBuildPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // The e-graph is per-compilation state; building here validates that
        // import succeeds and reports class statistics (the saturate pass
        // constructs its own instance over the same graph deterministically).
        EGraphConfig cfg;
        cfg.maxNodes = constants::kEgraphDefaultMaxNodes;
        const MathDomainProfile& profile = *ctx.domainProfile;
        EGraph eg(profile, cfg);
        if (graph.outputs().empty()) {
            return err(ErrorCode::InvalidGraph,
                       "egraph.build requires at least one output", 47);
        }
        MLK_TRY_VAR(rootClass, eg.importGraph(graph, graph.outputs()[0]));
        (void)rootClass;
        r.changed = false;
        return r;
    }
};

class EgraphSaturatePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        EGraphConfig cfg;
        cfg.maxNodes = constants::kEgraphDefaultMaxNodes;
        cfg.maxIterations = ctx.budget.fixpointIterations;
        EGraph eg(*ctx.domainProfile, cfg);
        MLK_TRY_VAR(rootClass, eg.importGraph(graph, graph.outputs()[0]));
        (void)rootClass;
        // Budgeted fixpoint (Rule 10): saturate until no growth or budget.
        for (uint32_t iter = 0; iter < cfg.maxIterations; ++iter) {
            if (ctx.cancel != nullptr && ctx.cancel->cancelled()) {
                return err(ErrorCode::Cancelled, "saturate cancelled", 132);
            }
            MLK_TRY_VAR(grew, eg.saturateOnce(*ctx.symbols));
            if (!grew) break;
        }
        r.changed = false;  // non-destructive: e-graph keeps all forms
        return r;
    }
};

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

        // Transplant with explicit id mapping (leaves first, then one result
        // per node in creation order — matches MathGraph allocation).
        const uint32_t base = graph.numValues();
        const uint32_t numLeaves = [&] {
            uint32_t count = 0;
            for (const auto& v : extracted.values()) {
                if (v.kind != ValueKind::NodeResult) ++count;
            }
            return count;
        }();
        for (const auto& v : extracted.values()) {
            switch (v.kind) {  // Rule 78: exhaustive
                case ValueKind::Constant:
                    if (v.constant.isInt) {
                        (void)graph.addIntConstant(v.constant.i64, v.type);
                    } else {
                        (void)graph.addConstant(v.constant.f64, v.type);
                    }
                    break;
                case ValueKind::Variable:
                    (void)graph.addVariable(v.name, v.type);
                    break;
                case ValueKind::Placeholder:
                    (void)graph.addPlaceholder(v.name, v.type);
                    break;
                case ValueKind::Symbol:
                    (void)graph.addSymbol(v.name, v.type);
                    break;
                case ValueKind::NodeResult:
                    break;  // recreated via addNode below
            }
        }
        SmallVector<ValueId, 8> mappedResults;
        ValueId extractedRootNew = kInvalidValueId;
        for (const auto& n : extracted.nodes()) {
            SmallVector<ValueId, 4> ins;
            for (const ValueId in : n.inputs) {
                const ValueId mapped = in < numLeaves
                                           ? static_cast<ValueId>(base + in)
                                           : mappedResults[in - numLeaves];
                ins.push_back(mapped);
            }
            MLK_TRY_VAR(newV, graph.addNode(n.op, ins, n.attrs));
            mappedResults.push_back(newV);
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

void registerEgraphPasses(SymbolTable& symbols) {
    static EgraphBuildPass build(symbols, "egraph.build", PassKind::Transform);
    static EgraphSaturatePass saturate(symbols, "egraph.saturate",
                                       PassKind::Transform);
    static EgraphExtractPass extract(symbols, "egraph.extract",
                                     PassKind::Transform);
    registerPass(symbols, build, PassKind::Transform, {"ir.verified"},
                 {"egraph.built"}, {}, {kEgraphTiers[0], kEgraphTiers[1]});
    registerPass(symbols, saturate, PassKind::Transform, {"egraph.built"},
                 {"egraph.saturated"}, {}, {kEgraphTiers[0], kEgraphTiers[1]});
    registerPass(symbols, extract, PassKind::Transform, {"egraph.saturated"},
                 {"math.extracted"}, {"analysis.cse", "analysis.cost"},
                 {kEgraphTiers[0], kEgraphTiers[1]});
}

}  // namespace mlk::passes
