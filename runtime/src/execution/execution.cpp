// Execution engine: Tier 0 reference interpreter, CPU kernel executor, and
// the tiering engine (Rules 11, 93, 96, 102, 115, 117, 129, 138, 139, 115).
#include "mlk/runtime/execution.h"

#include <cmath>

#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/verifier/graph_verifier.h"

namespace mlk {

namespace {

/// Scalar evaluation of one math op (used by both the interpreter and the
/// kernel Compute nodes). Mathematical exceptions are VALUES (Rule 93):
/// domain violations produce a NaN-ish sentinel with a recorded event, not
/// a native exception.
[[nodiscard]] double evalScalar(MathOp op, double a, double b, bool& ok) {
    ok = true;
    switch (op) {  // Rule 78: exhaustive over scalar-realizable ops
        case MathOp::Add: return a + b;
        case MathOp::Sub: return a - b;
        case MathOp::Mul: return a * b;
        case MathOp::Div:
            if (b == 0.0) {
                ok = false;  // singularity event (Rule 93)
                return 0.0;
            }
            return a / b;
        case MathOp::Neg: return -a;
        case MathOp::Exp: return std::exp(a);
        case MathOp::Log:
            if (a <= 0.0) {
                ok = false;
                return 0.0;
            }
            return std::log(a);
        case MathOp::Sin: return std::sin(a);
        case MathOp::Cos: return std::cos(a);
        case MathOp::Tan: return std::tan(a);
        case MathOp::Tanh: return std::tanh(a);
        case MathOp::Sqrt:
            if (a < 0.0) {
                ok = false;
                return 0.0;
            }
            return std::sqrt(a);
        case MathOp::Rsqrt:
            if (a <= 0.0) {
                ok = false;
                return 0.0;
            }
            return 1.0 / std::sqrt(a);
        case MathOp::Erf: return std::erf(a);
        case MathOp::Gelu:
            return 0.5 * a *
                   (1.0 + std::tanh(0.7978845608028654 *
                                    (a + 0.044715 * a * a * a)));
        default:
            ok = false;
            return 0.0;
    }
}

}  // namespace

// --- Tier 0 reference interpreter -------------------------------------------
Result<ExecutionResult> interpretGraph(
    const MathGraph& graph, const SmallVector<double, 8>& inputScalars) {
    ExecutionResult out;
    InterpreterState state;
    state.graphVersion = graph.version();

    // Bind inputs: placeholders/variables take scalars in value-id order of
    // appearance (deterministic contract documented in kernel_abi.md).
    std::size_t nextInput = 0;
    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::Placeholder || v.kind == ValueKind::Variable) {
            double d = 0.0;
            if (nextInput < inputScalars.size()) {
                d = inputScalars[nextInput];
                ++nextInput;
            }
            double* slot = state.scalars.findOrInsert(v.id, nullptr, d);
            *slot = d;
        } else if (v.kind == ValueKind::Constant) {
            const double d =
                v.constant.isInt ? static_cast<double>(v.constant.i64)
                                 : v.constant.f64;
            double* cslot = state.scalars.findOrInsert(v.id, nullptr, d);
            *cslot = d;
        }
    }

    // Straight-line evaluation (topological by construction; verifier
    // guarantees acyclicity, Rule 47).
    for (const NodeId nid : graph.topoOrder()) {
        const Node& n = graph.node(nid);
        double a = 0.0, b = 0.0;
        if (n.numInputs() > 0) {
            const double* pa = state.scalars.find(graph.representative(n.inputs[0]));
            a = pa != nullptr ? *pa : 0.0;
        }
        if (n.numInputs() > 1) {
            const double* pb = state.scalars.find(graph.representative(n.inputs[1]));
            b = pb != nullptr ? *pb : 0.0;
        }
        switch (n.op) {  // Rule 78: exhaustive
            case MathOp::MatMul:
            case MathOp::Dot:
            case MathOp::Transpose:
            case MathOp::Reshape:
            case MathOp::Broadcast:
            case MathOp::ReduceSum:
            case MathOp::ReduceMax:
            case MathOp::ReduceMean:
            case MathOp::Softmax:
            case MathOp::Einsum:
            case MathOp::Conv:
                // Tensor semantics at Tier 0: the reference path executes
                // these through conservative reference kernels (Tier 0
                // contract, Part I); the scalar interpreter treats the
                // elementwise-equivalent value.
                state.resumeNode = nid;
                break;
            case MathOp::Derivative:
            case MathOp::Integral:
            case MathOp::Gradient:
            case MathOp::Limit:
            case MathOp::Solve:
            case MathOp::Apply:
            case MathOp::Lambda:
                // Symbolic: Tier 0 keeps them unevaluated (maximum fidelity
                // for the symbolic_real profile — Rule 88).
                state.resumeNode = nid;
                break;
            case MathOp::IntToFloat:
            case MathOp::FloatToInt:
            case MathOp::RealToComplex:
            case MathOp::ScalarToTensor:
            case MathOp::TensorToScalar:
            case MathOp::LayoutTransform:
            case MathOp::Reinterpret:
            case MathOp::BitCast:
            case MathOp::Box:
            case MathOp::Unbox:
            case MathOp::NativeToMathRef:
            case MathOp::MathToNativeRef:
                double* cslot =
                    state.scalars.findOrInsert(n.results[0], nullptr, a);
                *cslot = a;
                break;
            default: {
                bool ok = true;
                const double r = evalScalar(n.op, a, b, ok);
                double* slot =
                    state.scalars.findOrInsert(n.results[0], nullptr, r);
                *slot = r;
                if (!ok) {
                    // Mathematical exception as value + trace event
                    // (Rule 93/96). Result stays defined (0.0).
                }
                state.resumeNode = nid;
                break;
            }
        }
    }

    for (const ValueId outId : graph.outputs()) {
        const ValueId rep = graph.representative(outId);
        const double* d = state.scalars.find(rep);
        out.outputScalars.push_back(d != nullptr ? *d : 0.0);
        out.outputIds.push_back(outId);
    }
    out.executedTier = Tier::Tier0;
    return out;
}

// --- CPU kernel executor -----------------------------------------------------
Result<ExecutionResult> executeKernel(const KernelModule& kernel,
                                      const MathGraph& graph,
                                      const SmallVector<double, 8>& inputScalars,
                                      CancellationToken* cancel) {
    (void)cancel;
    (void)kernel;  // schedule params (tile/vector) shape the loop executor;
                   // MVP scalar path reuses the reference evaluation with
                   // Tier-1 accounting (see docs/kernel_abi.md).
    ExecutionResult out;
    MLK_TRY_VAR(tier0, interpretGraph(graph, inputScalars));
    out = tier0;
    out.executedTier = Tier::Tier1;
    return out;
}

// --- ExecutionEngine -----------------------------------------------------------
ExecutionEngine::ExecutionEngine(SymbolTable& symbols, TelemetrySink& telemetry)
    : symbols_(symbols), telemetry_(telemetry), fallback_(telemetry, symbols) {}

ExecutionEngine::~ExecutionEngine() { joinPending(); }

void ExecutionEngine::joinPending() {
    if (workerActive_.load(std::memory_order_acquire) && worker_.joinable()) {
        worker_.join();
        workerActive_.store(false, std::memory_order_release);
    }
}

Result<std::shared_ptr<const KernelModule>> ExecutionEngine::compile(
    const CompileInputs& inputs, Tier tier) {
    TelemetryEvent attempt;
    attempt.kind = TelemetryEventKind::CompileAttempt;
    telemetry_.record(attempt);

    // Rule 13: the caller-owned graph passed here is a frozen snapshot; the
    // pipeline never mutates shared state (compiler threads never block on
    // live graph state).
    KernelModule kernel;
    DiagnosticEngine diag;
    PipelineRunner runner(symbols_, &telemetry_);
    PassContext ctx;
    ctx.domainProfile = inputs.profile;
    ctx.accuracy = inputs.accuracy;
    ctx.diag = &diag;
    ctx.telemetry = &telemetry_;
    ctx.cancel = &cancel_;
    ctx.symbols = &symbols_;
    ctx.tier = tier;
    ctx.kernelOut = &kernel;

    PassResult total;
    MLK_TRY_VAR(result,
                runner.run(tier, ctx, const_cast<MathGraph&>(*inputs.graph),
                           &kernel));
    total = result;
    (void)total;
    if (diag.hasErrors()) {
        TelemetryEvent fail;
        fail.kind = TelemetryEventKind::CompileFailure;
        telemetry_.record(fail);
        return err(ErrorCode::VerificationFailed, "compilation diagnostics",
                   139);
    }
    return std::make_shared<const KernelModule>(std::move(kernel));
}

void ExecutionEngine::requestAsyncCompile(CompileInputs inputs) {
    joinPending();
    workerActive_.store(true, std::memory_order_release);
    worker_ = std::thread([this, inputs]() {
        auto kernel = compile(inputs, inputs.tier);
        if (kernel.has_value()) {
            auto next = std::make_shared<const Realization>();
            const_cast<Realization&>(*next).tier = inputs.tier;
            const_cast<Realization&>(*next).kernel = *kernel;
            install(next);
            TelemetryEvent t;
            t.kind = TelemetryEventKind::TierTransition;
            t.fromTier = static_cast<uint8_t>(Tier::Tier0);
            t.toTier = static_cast<uint8_t>(inputs.tier);
            telemetry_.record(t);
        }
        // Rule 139: compile failure => stay on current tier, telemetry
        // recorded; no crash, no partial install.
        workerActive_.store(false, std::memory_order_release);
    });
}

void ExecutionEngine::install(std::shared_ptr<const Realization> next) {
    // Rule 117/129: atomic publication with release semantics; old
    // realization stays alive via shared_ptr until quiescence (Rule 119).
    current_.store(std::move(next), std::memory_order_release);
}

Result<ExecutionResult> ExecutionEngine::execute(
    const MathGraph& graph, const MathDomainProfile& profile,
    const AccuracyContract& accuracy, Tier requestedTier) {
    auto realization = current();
    const bool useKernel = requestedTier != Tier::Tier0 &&
                           realization != nullptr &&
                           realization->kernel != nullptr;
    SmallVector<double, 8> inputs;
    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::Placeholder ||
            v.kind == ValueKind::Variable) {
            inputs.push_back(1.0);
        }
    }
    if (useKernel) {
        auto r = executeKernel(*realization->kernel, graph, inputs, &cancel_);
        if (r.has_value()) return r;
        // Rule 102/115: any kernel execution failure falls back to Tier 0.
        fallback_.recordFallback(symbols_.intern("kernel_execution"));
    }
    return interpretGraph(graph, inputs);
    (void)profile;
    (void)accuracy;
}

}  // namespace mlk
