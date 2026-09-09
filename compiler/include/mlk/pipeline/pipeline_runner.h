// MLK+ pipeline runner (spec §9: per-tier pipelines; Rules 10, 47, 60, 143).
#pragma once

#include "mlk/core/cancellation.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/pass/pass.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/support/kernel_ir.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

/// Per-tier pipeline definitions (spec §9.1-9.4). Pass names in execution
/// order; missing passes (not yet registered) are a configuration error.
[[nodiscard]] SmallVector<SymbolId, 16> tierPipeline(Tier tier,
                                                     SymbolTable& symbols);

struct PipelineRunOptions {
    bool verifyBetweenPasses{true};  // Rule 47 in debug/test builds
    bool recordTelemetry{true};
};

/// Runs a tier's pipeline over a frozen graph snapshot. Deterministic
/// (Rule 143); honors kill switches (Rule 60/150); bounded budgets
/// (Rule 131); verifier between passes (Rule 47).
class PipelineRunner {
public:
    PipelineRunner(SymbolTable& symbols, IEventSink* telemetry);

    [[nodiscard]] Result<PassResult> run(Tier tier, PassContext& baseCtx,
                                         MathGraph& graph,
                                         KernelModule* kernelOut,
                                         const PipelineRunOptions& opts =
                                             PipelineRunOptions{});

private:
    SymbolTable& symbols_;
    IEventSink* telemetry_;
};

}  // namespace mlk
