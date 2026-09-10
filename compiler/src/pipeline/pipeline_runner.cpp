// Pipeline implementation: tier pipelines per spec §9.
#include "mlk/pipeline/pipeline_runner.h"

#include "mlk/core/event_sink.h"
#include "mlk/cost/cost_model.h"
#include "mlk/verifier/graph_verifier.h"

namespace mlk {

namespace {
struct TierPipelineDef {
    const char* passes[24];
};
}  // namespace

SmallVector<SymbolId, 16> tierPipeline(Tier tier, SymbolTable& symbols) {
    // Spec §9 pipelines, restricted to the registered MVP set (layout spec
    // §11 sanctions starting here; the full catalog is in pass_registry.md).
    SmallVector<SymbolId, 16> out;
    auto add = [&](const char* n) { out.push_back(symbols.intern(n)); };
    switch (tier) {  // Rule 78: exhaustive
        case Tier::Tier0:
            add("ir.verify");
            add("type.infer");
            add("shape.infer");
            add("property.infer");
            add("effect.infer");
            break;
        case Tier::Tier1:
            add("ir.verify");
            add("type.infer");
            add("shape.infer");
            add("property.infer");
            add("effect.infer");
            add("math.canonicalize");
            add("math.dce");
            add("math.cse");
            add("tensor.layout_infer");
            add("tensor.fusion_find");
            add("schedule.region_extract");
            add("schedule.tile");
            add("schedule.vectorize");
            add("schedule.parallelize");
            add("memory.buffer_plan");
            add("memory.place");
            add("memory.layout_select");
            add("lower.to_kernel_ir");
            add("backend.emit_binary");
            break;
        case Tier::Tier2:
            add("ir.verify");
            add("type.infer");
            add("shape.infer");
            add("property.infer");
            add("effect.infer");
            add("alias.infer");
            add("accuracy.analyze");
            add("cost.roofline");
            add("math.canonicalize");
            add("math.cse");
            add("math.dce");
            add("approx.policy_gate");
            add("approx.function_lower");
            add("approx.ulp_verify");
            add("calculus.derivative_symbolic");
            add("calculus.derivative_simplify");
            add("tensor.einsum_lower");
            add("tensor.transpose_elim");
            add("tensor.matmul_algorithm_select");
            add("tensor.fusion_find");
            add("schedule.region_extract");
            add("schedule.tile");
            add("schedule.vectorize");
            add("schedule.parallelize");
            add("memory.buffer_plan");
            add("memory.place");
            add("memory.layout_select");
            add("lower.to_kernel_ir");
            add("backend.emit_binary");
            break;
        case Tier::Tier3:
            // compile=INF: full analysis + e-graph saturation + proofs.
            add("ir.verify");
            add("type.infer");
            add("shape.infer");
            add("property.infer");
            add("effect.infer");
            add("alias.infer");
            add("accuracy.analyze");
            add("cost.roofline");
            add("math.canonicalize");
            add("math.cse");
            add("math.dce");
            add("egraph.build");
            add("egraph.saturate");
            add("egraph.extract");
            add("approx.policy_gate");
            add("approx.function_lower");
            add("approx.ulp_verify");
            add("calculus.derivative_symbolic");
            add("calculus.derivative_simplify");
            add("tensor.einsum_lower");
            add("tensor.transpose_elim");
            add("tensor.contraction_path");
            add("tensor.matmul_algorithm_select");
            add("tensor.fusion_find");
            add("schedule.region_extract");
            add("schedule.tile");
            add("schedule.vectorize");
            add("schedule.parallelize");
            add("memory.buffer_plan");
            add("memory.place");
            add("memory.layout_select");
            add("lower.to_kernel_ir");
            add("backend.emit_binary");
            break;
    }
    return out;
}

PipelineRunner::PipelineRunner(SymbolTable& symbols, IEventSink* telemetry)
    : symbols_(symbols), telemetry_(telemetry) {}

Result<PassResult> PipelineRunner::run(Tier tier, PassContext& baseCtx,
                                       MathGraph& graph,
                                       KernelModule* kernelOut,
                                       const PipelineRunOptions& opts) {
    // Analysis passes in the Tier 2/3 pipelines (cost.roofline) require a
    // cost model. When the embedding tool provides none, the runner installs
    // its own deterministic table-driven model instead of failing the whole
    // compilation (Rule 139: compiler config gaps degrade, never crash;
    // Rule 27: model internals are named constants). Callers that set
    // ctx.costModel explicitly keep full control.
    if (baseCtx.costModel == nullptr) {
        baseCtx.costModel = &defaultCostModel_;
    }
    PassResult total;
    total.nodesBefore = graph.liveNodeCount();
    const auto passNames = tierPipeline(tier, symbols_);
    for (const SymbolId nameId : passNames) {
        if (baseCtx.cancel != nullptr && baseCtx.cancel->cancelled()) {
            return err(ErrorCode::Cancelled, "pipeline cancelled", 132);
        }
        Pass* pass = PassRegistry::instance().byName(nameId);
        if (pass == nullptr) {
            return err(ErrorCode::Unimplemented,
                       "pipeline references unregistered pass: " +
                           symbols_.text(nameId));
        }
        if (baseCtx.killed(nameId)) {
            // Kill switch honored (Rule 60/150): event recorded, pass
            // skipped, pipeline continues (bisection support).
            if (telemetry_ != nullptr && opts.recordTelemetry) {
                telemetry_->event(10, nameId,
                                  symbols_.intern("kill_switch_skipped"), 1);
            }
            continue;
        }
        // Contract tier check (Rule 142).
        const PassContract* contract =
            PassRegistry::instance().contractByName(nameId);
        if (contract != nullptr) {
            bool tierOk = false;
            for (const Tier t : contract->supportedTiers) {
                tierOk = tierOk || t == tier;
            }
            if (!tierOk) {
                return err(ErrorCode::UnsupportedCapability,
                           "pass " + symbols_.text(nameId) +
                               " does not support " + tierName(tier) +
                               " (Rule 142)");
            }
        }
        PassContext ctx = baseCtx;
        ctx.kernelOut = (nameId == symbols_.intern("lower.to_kernel_ir") ||
                         nameId == symbols_.intern("backend.emit_binary"))
                            ? kernelOut
                            : nullptr;
        MLK_TRY_VAR(result, pass->run(ctx, graph));
        total.changed = total.changed || result.changed;
        for (const auto& inv : result.invalidatedAnalyses) {
            total.invalidatedAnalyses.push_back(inv);
        }
        // Rule 47: verifier runs between passes (all builds here; debug
        // builds enforce via verifyBetweenPasses default true).
        if (opts.verifyBetweenPasses) {
            DiagnosticEngine diag;
            VerifyOptions vopts;
            if (!verifyGraph(graph, baseCtx.domainProfile, vopts, diag)) {
                for (const auto& d : diag.entries()) baseCtx.diag->report(d);
                return err(ErrorCode::VerificationFailed,
                           "post-pass verification failed after " +
                               symbols_.text(nameId) + " (Rule 47)",
                           47);
            }
        }
        // Rule 30/138: no silent transitions — every failure is recorded by
        // the caller via telemetry sink; here we record pass outcomes.
        if (telemetry_ != nullptr && opts.recordTelemetry) {
            telemetry_->event(11, nameId, kInvalidSymbolId,
                              result.changed ? 1 : 0);
        }
    }
    total.nodesAfter = graph.liveNodeCount();
    return total;
}

}  // namespace mlk
