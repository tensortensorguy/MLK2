// MLK+ execution engine (Rule 11: evaluation threads never block on JIT;
// Rule 115: Tier 0 is the universal fallback; Rule 117: atomic publication;
// Rule 129: safe kernel pointer swaps; Rule 138: tier transitions observable).
#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "mlk/core/cancellation.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/pass/pass.h"
#include "mlk/runtime/buffer.h"
#include "mlk/runtime/fallback.h"
#include "mlk/runtime/safepoint.h"
#include "mlk/runtime/telemetry.h"

namespace mlk {

class PipelineRunner;

/// A published realization: either an executable kernel module (Tier>=1) or
/// the Tier 0 interpreter path. Publication is atomic with release
/// semantics; consumers acquire (Rule 117/129).
struct Realization {
    Tier tier{Tier::Tier0};
    std::shared_ptr<const KernelModule> kernel;  // null => Tier 0 path
};

/// Output of a graph execution (scalar results keyed by output id).
struct ExecutionResult {
    SmallVector<double, 8> outputScalars{};
    SmallVector<ValueId, 8> outputIds{};
    Tier executedTier{Tier::Tier0};
};

/// Compilation job inputs (Rule 143: determinism requires the full tuple).
struct CompileInputs {
    const MathGraph* graph{nullptr};
    const MathDomainProfile* profile{nullptr};
    const AccuracyContract* accuracy{nullptr};
    Tier tier{Tier::Tier1};
};

/// ExecutionEngine: runs graphs; compiles higher tiers on background
/// threads; installs via atomic pointer swap; falls back to Tier 0 on any
/// failure (Rule 139).
class ExecutionEngine {
public:
    ExecutionEngine(SymbolTable& symbols, TelemetrySink& telemetry);
    ~ExecutionEngine();

    /// Synchronous compile of a snapshot (Rule 13: frozen snapshot).
    [[nodiscard]] Result<std::shared_ptr<const KernelModule>> compile(
        const CompileInputs& inputs, Tier tier);

    /// Asynchronous tier-up (Rule 11): spawns background compilation; the
    /// current tier keeps executing; installation is atomic.
    void requestAsyncCompile(CompileInputs inputs);

    /// Executes: Tier 0 interpreter for tier==Tier0 or when no kernel is
    /// installed yet; otherwise the kernel executor (with Tier 0 fallback).
    /// `inputScalars` binds placeholders/variables in value-id order
    /// (documented ABI order; see kernel_abi.md).
    [[nodiscard]] Result<ExecutionResult> execute(
        const MathGraph& graph, const MathDomainProfile& profile,
        const AccuracyContract& accuracy, Tier requestedTier,
        const SmallVector<double, 8>& inputScalars);

    [[nodiscard]] std::shared_ptr<const Realization> current() const noexcept {
        return current_.load(std::memory_order_acquire);
    }

    [[nodiscard]] CancellationToken& cancelToken() noexcept { return cancel_; }

private:
    void install(std::shared_ptr<const Realization> next);
    void joinPending();

    SymbolTable& symbols_;
    TelemetrySink& telemetry_;
    CancellationToken cancel_;
    std::atomic<std::shared_ptr<const Realization>> current_{};
    std::thread worker_;
    std::atomic<bool> workerActive_{false};
    FallbackEngine fallback_;
};

/// Tier 0 reference interpreter: maximum semantic fidelity (Part I tier 0).
/// Executes the math graph directly; mathematical exceptions are values
/// (Rule 93), tracing hooks fire per node (Rule 96).
[[nodiscard]] Result<ExecutionResult> interpretGraph(
    const MathGraph& graph, const SmallVector<double, 8>& inputScalars);

/// Kernel executor (CPU): tree-walks a KernelModule over buffers. No dynamic
/// codegen (see docs/kernel_abi.md for the W^X publication note).
[[nodiscard]] Result<ExecutionResult> executeKernel(
    const KernelModule& kernel, const MathGraph& graph,
    const SmallVector<double, 8>& inputScalars,
    CancellationToken* cancel);

/// --- Buffer-level kernel ABI (docs/kernel_abi.md) -------------------------
/// Raw element-buffer bindings for tensor kernels. Inputs are bound in
/// graph placeholder/variable value-id order; outputs in graph output
/// order; scalar params in the kernel's ScalarParam first-touch order.
/// All buffers are dense contiguous f64 rows (row-major for 2-D).
struct KernelBufferBindings {
    SmallVector<double*, 8> inputs{};
    SmallVector<double*, 8> outputs{};
    SmallVector<double, 8> scalars{};
    /// Element count for dynamic loop bounds (kKernelLoopDynamicBound).
    int64_t elements{0};
    /// Executor-allocated scratch for isTemp buffers (bufferId-indexed;
    /// null for every non-temp id). Zero-initialized; filled by
    /// executeKernelOnBuffers before execution, never by external
    /// callers. Every element must be written before it is read (the
    /// synthesized init statements guarantee that).
    SmallVector<double*, 8> temps{};
};

/// Executes a KernelModule directly over dense f64 buffers: fused
/// elementwise compute chains (with implementation-family dispatch:
/// libm / verified poly7 sin), threaded loops per the schedule params,
/// and blocked GEMM Call nodes with declarative tile params. Cancellation
/// is polled per parallel chunk (Rule 107/130: bounded latency).
[[nodiscard]] Result<void> executeKernelOnBuffers(
    const KernelModule& kernel, SymbolTable& symbols,
    const KernelBufferBindings& io, CancellationToken* cancel);

}  // namespace mlk
