// MLK+ buffer-level kernel executor (see docs/kernel_abi.md).
//
// Executes KernelModules directly over dense f64 element buffers:
//   - Loop { Compute(expr chain); Store } for fused elementwise kernels,
//     with implementation-family dispatch (libm default; "poly7" = the
//     superoptimizer-verified sin family in mlk/support/math_families.h —
//     Rule 50/58: verified before use, Rule 62: family without a verified
//     implementation falls back to libm and is reported via telemetry),
//   - Call(MathOp::MatMul) as a blocked row-major GEMM with declarative
//     tile params (Rule 54),
//   - threading over disjoint chunks; the executor owns the thread-count
//     decision (schedule.parallelize contract), bounded cancellation.
//
// This is its own translation unit — one executor per file, mirroring the
// one-file-per-pass hygiene of compiler/src/passes.
#include "mlk/runtime/execution.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "mlk/core/constants.h"
#include "mlk/support/math_families.h"

namespace mlk {

namespace {

/// Hot-path knobs (Rule 27: named, documented constants only).
/// kKernelExecMaxThreads / kParallelChunkElements live in constants.h —
/// the fast-kernel launch-coverage certificate mirrors the chunking
/// arithmetic and must read the SAME values.
inline constexpr std::size_t kMaxTempSlots = 64;
/// Upper bound on ONE temp buffer's element count (DoS guard for
/// executor-allocated scratch; 16M doubles = 128 MiB).
inline constexpr int64_t kKernelTempElementsLimit = 1 << 24;

enum class SinFamily { Libm, Poly7 };

[[nodiscard]] SinFamily resolveFamily(SymbolTable& symbols,
                                      SymbolId family) noexcept {
    if (family == kInvalidSymbolId) return SinFamily::Libm;
    if (symbols.text(family) == "poly7") {
        // "poly7" is verified for Sin only (math_families.h certificate);
        // other ops silently keep the libm family (Rule 62 fallback).
        return SinFamily::Poly7;
    }
    return SinFamily::Libm;
}

/// Per-element evaluator for a fused expression chain. Temps are SSA
/// results of prior exprs; the last expr's value is the compute result.
struct ExprEvaluator {
    const SmallVector<KernelExpr, 8>& exprs;
    const double* elemA;
    const double* elemB;
    const SmallVector<double, 8>& scalars;
    SinFamily sinFamily;
    double temps[kMaxTempSlots] = {};
    // Polyhedral multi-dim context: enclosing loop-var values (outermost
    // first) and the buffer table for ElemIdx operands. Empty stack and
    // null module => legacy 1-D evaluation.
    const SmallVector<int64_t, 8>* vars{nullptr};
    const KernelModule* module{nullptr};
    const KernelBufferBindings* io{nullptr};

    /// Flat element address of an ElemIdx operand:
    ///   sum coeffs[p] * vars[p] + offset
    [[nodiscard]] int64_t flatIndex(const KernelOperand& o) const noexcept {
        int64_t flat = o.idxOffset;
        if (vars != nullptr) {
            const std::size_t n = o.idxCoeffs.size() < vars->size()
                                      ? o.idxCoeffs.size()
                                      : vars->size();
            for (std::size_t p = 0; p < n; ++p) {
                flat += o.idxCoeffs[p] * (*vars)[p];
            }
        }
        return flat;
    }

    /// Resolves an ElemIdx operand to its buffer element pointer.
    [[nodiscard]] const double* elemPtr(const KernelOperand& o) const {
        if (module == nullptr || io == nullptr) return nullptr;
        if (o.index < 0 ||
            static_cast<uint32_t>(o.index) >= module->buffers.size()) {
            return nullptr;
        }
        const uint32_t bid = static_cast<uint32_t>(o.index);
        const double* base = nullptr;
        if (module->buffers[bid].isTemp) {
            base = bid < io->temps.size() ? io->temps[bid] : nullptr;
        } else if (module->buffers[bid].isInput) {
            base = bid < io->inputs.size() ? io->inputs[bid] : nullptr;
        } else {
            const uint32_t outIdx =
                bid - static_cast<uint32_t>(io->inputs.size());
            base = outIdx < io->outputs.size() ? io->outputs[outIdx]
                                               : nullptr;
        }
        return base != nullptr ? base + flatIndex(o) : nullptr;
    }

    [[nodiscard]] double evalOne(const KernelOperand& o,
                                 const int64_t i) const noexcept {
        switch (o.kind) {  // Rule 78: exhaustive
            case KernelOperand::Kind::Const: return o.constValue;
            case KernelOperand::Kind::ElemA: return elemA[i];
            case KernelOperand::Kind::ElemB: return elemB != nullptr
                                                    ? elemB[i]
                                                    : 0.0;
            case KernelOperand::Kind::ScalarParam: {
                const auto idx = static_cast<std::size_t>(o.index);
                return idx < scalars.size() ? scalars[idx] : 0.0;
            }
            case KernelOperand::Kind::Temp: {
                const auto idx = static_cast<std::size_t>(o.index);
                return idx < kMaxTempSlots ? temps[idx] : 0.0;
            }
            case KernelOperand::Kind::ElemIdx: {
                const double* p = elemPtr(o);
                return p != nullptr ? *p : 0.0;
            }
        }
        return 0.0;
    }

    [[nodiscard]] double apply(MathOp op, double a, double b) const noexcept {
        switch (op) {  // Rule 78: exhaustive over scalar-realizable ops
            case MathOp::Add: return a + b;
            case MathOp::Sub: return a - b;
            case MathOp::Mul: return a * b;
            case MathOp::Div: return b != 0.0 ? a / b : 0.0;
            case MathOp::Neg: return -a;
            case MathOp::Pow: return a == 2.0 && b == 2.0 ? a * a
                                                          : std::pow(a, b);
            case MathOp::Exp: return std::exp(a);
            case MathOp::Log: return a > 0.0 ? std::log(a) : 0.0;
            case MathOp::Sin:
                return sinFamily == SinFamily::Poly7 ? families::polySin(a)
                                                     : std::sin(a);
            case MathOp::Cos: return std::cos(a);
            case MathOp::Tan: return std::tan(a);
            case MathOp::Tanh: return std::tanh(a);
            case MathOp::Sqrt: return a >= 0.0 ? std::sqrt(a) : 0.0;
            case MathOp::Rsqrt:
                return a > 0.0 ? 1.0 / std::sqrt(a) : 0.0;
            case MathOp::Erf: return std::erf(a);
            case MathOp::Gelu:
                return 0.5 * a * (1.0 + std::tanh(
                    0.7978845608028654 * (a + 0.044715 * a * a * a)));
            default:
                // Non-realizable ops are rejected at lowering time
                // (isScalarRealizable); reaching here is a contract bug.
                return 0.0;
        }
    }

    [[nodiscard]] double run(const int64_t i) noexcept {
        for (std::size_t k = 0; k < exprs.size() && k < kMaxTempSlots; ++k) {
            const KernelExpr& e = exprs[k];
            const bool unary = (e.op == MathOp::Neg || e.op == MathOp::Exp ||
                                e.op == MathOp::Log || e.op == MathOp::Sin ||
                                e.op == MathOp::Cos || e.op == MathOp::Tan ||
                                e.op == MathOp::Tanh ||
                                e.op == MathOp::Sqrt ||
                                e.op == MathOp::Rsqrt ||
                                e.op == MathOp::Erf ||
                                e.op == MathOp::Gelu);
            const double a = evalOne(e.a, i);
            const double b = unary ? 0.0 : evalOne(e.b, i);
            temps[k] = apply(e.op, a, b);
        }
        const auto last = static_cast<std::size_t>(exprs.size()) - 1;
        return last < kMaxTempSlots ? temps[last] : 0.0;
    }
};

/// Decides the thread count for a loop (executor-owned per the
/// schedule.parallelize contract; Rule 137: no locks on this path).
[[nodiscard]] std::size_t decideThreads(const KernelModule& kernel,
                                        SymbolTable& symbols,
                                        const int64_t elements) {
    std::size_t threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    if (threads > constants::kKernelExecMaxThreads) {
        threads = constants::kKernelExecMaxThreads;
    }
    const SymbolId threadsKey = symbols.intern("threads");
    if (const int64_t* v = kernel.scheduleParams.find(threadsKey)) {
        if (*v >= 1) {
            threads = static_cast<std::size_t>(*v);
        }
    }
    if (elements < constants::kParallelChunkElements) threads = 1;
    return threads;
}

/// Kernel buffer ids index the module's combined buffer table (inputs
/// first, then outputs — the lowering's emission order). The ABI passes
/// inputs and outputs as separate arrays, so resolve through the table
/// layout instead of indexing both arrays with the raw id.
[[nodiscard]] const double* resolveInput(const KernelModule& kernel,
                                         const KernelBufferBindings& io,
                                         const uint32_t bid) noexcept {
    if (bid == constants::kInvalidId || bid >= kernel.buffers.size()) {
        return nullptr;
    }
    const KernelBuffer& b = kernel.buffers[bid];
    if (b.isInput) {
        return bid < io.inputs.size() ? io.inputs[bid] : nullptr;
    }
    const uint32_t outIdx = bid - static_cast<uint32_t>(io.inputs.size());
    return outIdx < io.outputs.size() ? io.outputs[outIdx] : nullptr;
}

[[nodiscard]] double* resolveOutput(const KernelModule& kernel,
                                    const KernelBufferBindings& io,
                                    const uint32_t bid) noexcept {
    if (bid == constants::kInvalidId || bid >= kernel.buffers.size()) {
        return nullptr;
    }
    const KernelBuffer& b = kernel.buffers[bid];
    if (b.isOutput) {
        const uint32_t outIdx = bid - static_cast<uint32_t>(io.inputs.size());
        return outIdx < io.outputs.size() ? io.outputs[outIdx] : nullptr;
    }
    return bid < io.inputs.size() ? io.inputs[bid] : nullptr;
}

/// Row-wise ReduceSum for Call(ReduceSum) nodes (shared by the 1-D path
/// and the multi-dim walker): Y[i] = sum_k A[i*K + k], accumulated in
/// ascending k — the exact order the polyhedral accumulate chain
/// preserves (Rule 90: bit-exact vs the transformed kernel).
[[nodiscard]] Result<void> execReduceSumCall(const KernelModule& kernel,
                                             const KernelBufferBindings& io,
                                             CancellationToken* cancel,
                                             const KernelNode& top) {
    if (top.bufferA >= kernel.buffers.size() ||
        top.bufferOut >= kernel.buffers.size()) {
        return err(ErrorCode::InvalidGraph,
                   "reducesum buffer id out of range");
    }
    const KernelBuffer& ba = kernel.buffers[top.bufferA];
    const KernelBuffer& by = kernel.buffers[top.bufferOut];
    if (ba.dims.size() != 2 || by.dims.size() != 1) {
        return err(ErrorCode::InvalidGraph,
                   "reducesum requires rank-2 input over rank-1 output");
    }
    const double* a = resolveInput(kernel, io, top.bufferA);
    double* y = resolveOutput(kernel, io, top.bufferOut);
    if (a == nullptr || y == nullptr) {
        return err(ErrorCode::InvalidArgument,
                   "reducesum buffers not bound by the caller");
    }
    const int64_t m = ba.dims[0];
    const int64_t k = ba.dims[1];
    if (by.dims[0] != m || k <= 0 || m <= 0) {
        return err(ErrorCode::InvalidGraph,
                   "reducesum operand shapes incompatible");
    }
    for (int64_t i = 0; i < m; ++i) {
        if (cancel != nullptr && cancel->cancelled()) {
            return err(ErrorCode::Cancelled,
                       "kernel execution cancelled", 130);
        }
        const double* row = a + i * k;
        double acc = 0.0;
        for (int64_t p = 0; p < k; ++p) {
            acc += row[p];
        }
        y[i] = acc;
    }
    return {};
}

/// Row-wise stable softmax for Call(Softmax) nodes (shared reference
/// kernel; Rule 90: this IS the order contract the polyhedral chain
/// must reproduce). Per row m, four ascending-k passes:
///   rowmax: r = -inf; r = (x[m,k] > r) ? x[m,k] : r
///   exp:    Y[m,k] = exp(x[m,k] - r)   (Sub then Exp — the exact chain
///           the synthesized statement evaluates)
///   sum:    s = 0.0; s += Y[m,k]
///   div:    Y[m,k] = (s != 0.0) ? Y[m,k] / s : 0.0
/// The exp pass materializes into the OUTPUT row (allocation-free); the
/// synthesized chain reads the same values from its materialized e-temp,
/// so both forms evaluate identical operations on identical values.
[[nodiscard]] Result<void> execSoftmaxCall(const KernelModule& kernel,
                                           const KernelBufferBindings& io,
                                           CancellationToken* cancel,
                                           const KernelNode& top) {
    if (top.bufferA >= kernel.buffers.size() ||
        top.bufferOut >= kernel.buffers.size()) {
        return err(ErrorCode::InvalidGraph,
                   "softmax buffer id out of range");
    }
    const KernelBuffer& ba = kernel.buffers[top.bufferA];
    const KernelBuffer& by = kernel.buffers[top.bufferOut];
    if (ba.dims.size() != 2 || by.dims.size() != 2) {
        return err(ErrorCode::InvalidGraph,
                   "softmax requires rank-2 operand and output");
    }
    const double* a = resolveInput(kernel, io, top.bufferA);
    double* y = resolveOutput(kernel, io, top.bufferOut);
    if (a == nullptr || y == nullptr) {
        return err(ErrorCode::InvalidArgument,
                   "softmax buffers not bound by the caller");
    }
    const int64_t m = ba.dims[0];
    const int64_t k = ba.dims[1];
    if (by.dims[0] != m || by.dims[1] != k || k <= 0 || m <= 0) {
        return err(ErrorCode::InvalidGraph,
                   "softmax operand shapes incompatible");
    }
    for (int64_t i = 0; i < m; ++i) {
        if (cancel != nullptr && cancel->cancelled()) {
            return err(ErrorCode::Cancelled,
                       "kernel execution cancelled", 130);
        }
        const double* row = a + i * k;
        double* yrow = y + i * k;
        double r = -INFINITY;
        for (int64_t p = 0; p < k; ++p) {
            r = (row[p] > r) ? row[p] : r;
        }
        for (int64_t p = 0; p < k; ++p) {
            yrow[p] = std::exp(row[p] - r);
        }
        double s = 0.0;
        for (int64_t p = 0; p < k; ++p) {
            s += yrow[p];
        }
        for (int64_t p = 0; p < k; ++p) {
            yrow[p] = s != 0.0 ? yrow[p] / s : 0.0;
        }
    }
    return {};
}

void runElementwiseRange(const KernelNode& compute, const KernelNode& store,
                         const KernelModule& kernel, SymbolTable& symbols,
                         const KernelBufferBindings& io,
                         const int64_t begin, const int64_t end,
                         CancellationToken* cancel) {
    const double* inA = resolveInput(kernel, io, compute.bufferA);
    const double* inB = resolveInput(kernel, io, compute.bufferB);
    double* out = resolveOutput(kernel, io, store.bufferOut);
    if (inA == nullptr || out == nullptr) return;  // unbound ABI: skip

    (void)kernel;
    const SinFamily fam =
        compute.family == kInvalidSymbolId
            ? SinFamily::Libm
            : resolveFamily(symbols, compute.family);

    // Fused chains evaluate the whole expression tree per element; legacy
    // single-op Compute nodes (empty exprs) use compute.math directly.
    if (!compute.exprs.empty()) {
        ExprEvaluator ev{compute.exprs, inA, inB, io.scalars, fam};
        for (int64_t i = begin; i < end; ++i) {
            if (cancel != nullptr && cancel->cancelled()) return;
            out[i] = ev.run(i);
        }
        return;
    }
    ExprEvaluator ev{{}, inA, inB, io.scalars, fam};
    // Legacy path: one op over ElemA (+ElemB for binary ops).
    KernelOperand a;
    a.kind = KernelOperand::Kind::ElemA;
    KernelOperand b;
    b.kind = KernelOperand::Kind::ElemB;
    for (int64_t i = begin; i < end; ++i) {
        if (cancel != nullptr && cancel->cancelled()) return;
        const double va = ev.evalOne(a, i);
        const double vb = ev.evalOne(b, i);
        out[i] = ev.apply(compute.math, va, vb);
    }
}

/// Blocked row-major GEMM over one row-block range (thread-disjoint).
/// tileM slices the row range; tileN/tileK slice the column/K loops.
void gemmRowBlock(const double* a, const double* b, double* c,
                  const int64_t n, const int64_t k,
                  const int64_t tileM, const int64_t tileN,
                  const int64_t tileK,
                  const int64_t rowBegin, const int64_t rowEnd) {
    for (int64_t ii = rowBegin; ii < rowEnd; ii += tileM) {
        const int64_t iEnd = std::min(ii + tileM, rowEnd);
        for (int64_t jj = 0; jj < n; jj += tileN) {
            const int64_t jEnd = std::min(jj + tileN, n);
            for (int64_t kk = 0; kk < k; kk += tileK) {
                const int64_t kEnd = std::min(kk + tileK, k);
                for (int64_t i = ii; i < iEnd; ++i) {
                    const double* aRow = a + i * k;
                    double* cRow = c + i * n;
                    for (int64_t p = kk; p < kEnd; ++p) {
                        const double aik = aRow[p];
                        if (aik == 0.0) continue;
                        const double* bRow = b + p * n;
                        for (int64_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aik * bRow[j];
                        }
                    }
                }
            }
        }
    }
}


// --- Multi-dim tree walker (polyhedral kernels) -----------------------------
//
// Walks the KernelModule forest recursively with a loop-var stack:
//   - Loop: resolves bounds (affine beginCoeffs/endCoeffs over the stack,
//     buffer-dim bounds via endBuf/endDim, or constants) and iterates
//     var from begin while var < end (step must be 1 for multi-dim nests;
//     other steps are rejected at SCoP extraction),
//   - Compute: evaluates the fused expression chain with ElemIdx operands
//     and forwards the last temp to the paired Store,
//   - Store: writes (or accumulates into) the affine target,
//   - Call: delegated to the legacy blocked GEMM path.
// Parallel loops (KernelNode::parallel, set by poly.codegen from the
// scheduler's zero-distance rows) run disjoint index chunks on threads:
// the scheduler proved every dependence distance identically zero at
// that level, so slabs read/write disjoint locations and any interleaving
// produces the sequential result bit-exact (deterministic execution,
// Rule 43). Threads are bounded by the same executor-owned decision as
// the 1-D path (Rule 12/137); the chunking needs no locks.
struct MultiDimWalker {
    const KernelModule& kernel;
    SymbolTable& symbols;
    const KernelBufferBindings& io;
    CancellationToken* cancel;
    SmallVector<int64_t, 8> vars{};

    /// Thread outcome slot: without exceptions, each worker records its
    /// terminal state here; the parent joins and returns the first
    /// failure (deterministic diagnostics).
    struct Slot {
        bool failed{false};
        ErrorCode code{ErrorCode::Ok};
        uint32_t rule{0};
        std::string msg{};
    };

    /// True when every child subtree of the loop contains only Loop/
    /// Compute/Store nodes (the poly.codegen emission alphabet). Parallel
    /// chunking is withheld otherwise (Call nodes intern symbols — not a
    /// hot-path-safe operation under threading).
    [[nodiscard]] bool loopBodyThreadSafe(const KernelNode& loop) const {
        for (const uint32_t c : loop.children) {
            if (!subtreeThreadSafe(c)) return false;
        }
        return true;
    }

    /// True when the subtree rooted at nodeId contains only Loop/Compute/
    /// Store/Guard nodes (the poly.codegen emission alphabet). Guard
    /// predicates read the thread-local var stack only, so a guarded
    /// region is as thread-safe as its children. Parallel chunking is
    /// withheld for anything else (Call interns symbols — not a hot-
    /// path-safe operation under threading).
    [[nodiscard]] bool subtreeThreadSafe(uint32_t nodeId) const {
        if (nodeId >= kernel.nodes.size()) return false;
        const KernelNode& n = kernel.nodes[nodeId];
        switch (n.op) {  // Rule 78: exhaustive
            case KernelOp::Loop:
            case KernelOp::Compute:
            case KernelOp::Store:
            case KernelOp::Guard:
                break;
            default:
                return false;
        }
        for (const uint32_t c : n.children) {
            if (!subtreeThreadSafe(c)) return false;
        }
        return true;
    }

    /// Iterates one node list with Compute+Store pairing by sibling
    /// order (shared by loop bodies and guarded regions).
    [[nodiscard]] Result<void> execChildList(
        const SmallVector<uint32_t, 4>& children) {
        for (std::size_t ci = 0; ci < children.size(); ++ci) {
            const uint32_t cid = children[ci];
            if (cid >= kernel.nodes.size()) {
                return err(ErrorCode::InvalidGraph,
                           "walker: child id out of range");
            }
            const KernelNode& c = kernel.nodes[cid];
            if (c.op == KernelOp::Compute) {
                const KernelNode* store = nullptr;
                if (ci + 1 < children.size() &&
                    children[ci + 1] < kernel.nodes.size() &&
                    kernel.nodes[children[ci + 1]].op == KernelOp::Store) {
                    store = &kernel.nodes[children[ci + 1]];
                }
                MLK_TRYV(execNode(cid, store));
                if (store != nullptr) ++ci;  // consume Store
                continue;
            }
            MLK_TRYV(execNode(cid, nullptr));
        }
        return {};
    }

    /// Iterates one loop's body over the INCLUSIVE range [b, e], pushing
    /// each induction value onto the local var stack (the thread entry
    /// point for parallel chunks and the sequential path alike).
    [[nodiscard]] Result<void> execLoopRange(const KernelNode& n,
                                             const int64_t b,
                                             const int64_t e) {
        for (int64_t v = b; v <= e; ++v) {
            if (cancel != nullptr && cancel->cancelled()) {
                return err(ErrorCode::Cancelled,
                           "kernel execution cancelled", 130);
            }
            vars.push_back(v);
            // Children with Compute+Store pairing by sibling order.
            MLK_TRYV(execChildList(n.children));
            vars.pop_back();
        }
        return {};
    }

    /// Runs one parallel chunk on a fresh walker (own var stack seeded
    /// with the enclosing dims) and records the terminal state.
    static void runChunk(const KernelModule& kernel, SymbolTable& symbols,
                         const KernelBufferBindings& io,
                         CancellationToken* cancel,
                         const SmallVector<int64_t, 8>& outerVars,
                         const KernelNode& loop, const int64_t b,
                         const int64_t e, Slot& slot) {
        MultiDimWalker child{kernel, symbols, io, cancel, outerVars};
        auto r = child.execLoopRange(loop, b, e);
        if (!r.has_value()) {
            slot.failed = true;
            slot.code = r.error().code;
            slot.rule = r.error().rule;
            slot.msg = r.error().message;
        }
    }

    [[nodiscard]] Result<void> execNode(uint32_t nodeId,
                                        const KernelNode* pairedStore) {
        if (nodeId >= kernel.nodes.size()) {
            return err(ErrorCode::InvalidGraph,
                       "walker: node id out of range");
        }
        const KernelNode& n = kernel.nodes[nodeId];
        switch (n.op) {  // Rule 78: exhaustive
            case KernelOp::Loop: {
                if (n.step != 1) {
                    return err(ErrorCode::UnsupportedCapability,
                               "walker: non-unit loop step");
                }
                int64_t begin = n.begin;
                int64_t endInclusive = n.end - 1;
                if (!n.beginCoeffs.empty()) {
                    begin = n.beginOffset;
                    for (std::size_t p = 0;
                         p < n.beginCoeffs.size() && p < vars.size(); ++p) {
                        begin += n.beginCoeffs[p] * vars[p];
                    }
                }
                if (!n.endCoeffs.empty()) {
                    endInclusive = n.endOffset;
                    for (std::size_t p = 0;
                         p < n.endCoeffs.size() && p < vars.size(); ++p) {
                        endInclusive += n.endCoeffs[p] * vars[p];
                    }
                } else if (n.end == constants::kKernelLoopDynamicBound &&
                           n.endBuf != constants::kInvalidId &&
                           n.endDim >= 0 &&
                           n.endBuf < kernel.buffers.size() &&
                           n.endDim <
                               static_cast<int32_t>(
                                   kernel.buffers[n.endBuf].dims.size())) {
                    endInclusive =
                        kernel.buffers[n.endBuf].dims[n.endDim] - 1;
                } else if (n.end == constants::kKernelLoopDynamicBound) {
                    endInclusive = io.elements - 1;  // legacy dynamic form
                }
                if (endInclusive < begin) return {};  // empty loop
                // Parallel-marked loop: disjoint chunks on threads when
                // the trip is large enough to amortize the spawn and the
                // subtree is in the thread-safe emission alphabet.
                if (n.parallel && loopBodyThreadSafe(n)) {
                    const std::size_t threads =
                        decideThreads(kernel, symbols,
                                      endInclusive - begin + 1);
                    if (threads > 1) {
                        std::vector<Slot> slots(threads);
                        {
                            std::vector<std::thread> pool;
                            pool.reserve(threads);
                            const int64_t total =
                                endInclusive - begin + 1;
                            const int64_t chunk =
                                (total + static_cast<int64_t>(threads) -
                                 1) /
                                static_cast<int64_t>(threads);
                            for (std::size_t t = 0; t < threads; ++t) {
                                const int64_t b = begin +
                                    static_cast<int64_t>(t) * chunk;
                                const int64_t e =
                                    std::min(b + chunk - 1, endInclusive);
                                if (b > endInclusive) break;
                                pool.emplace_back(
                                    [this, &n, b, e, t, &slots]() {
                                        runChunk(kernel, symbols, io,
                                                 cancel, vars, n, b, e,
                                                 slots[t]);
                                    });
                            }
                            // joinPending is implicit: the destructor
                            // joins; explicit join keeps error flow clear.
                            for (auto& th : pool) th.join();
                        }
                        for (const Slot& s : slots) {
                            if (s.failed) {
                                return err(s.code, s.msg, s.rule);
                            }
                        }
                        return {};
                    }
                }
                MLK_TRYV(execLoopRange(n, begin, endInclusive));
                return {};
            }
            case KernelOp::Compute: {
                if (pairedStore == nullptr) {
                    // Bare Compute without its Store: no semantics here.
                    break;
                }
                MLK_TRYV(execPair(n, *pairedStore));
                break;
            }
            case KernelOp::Store:
            case KernelOp::Load:
            case KernelOp::AllocBuffer:
            case KernelOp::CopyBuffer:
            case KernelOp::Barrier:
            case KernelOp::Trace:
                // Consumed by the pairing above or carry no semantics
                // for this walker.
                break;
            case KernelOp::Guard: {
                if (!n.hasAffineGuard()) {
                    // Speculative guarded execution belongs to
                    // ExecutionEngine (Rule 5); nothing here.
                    break;
                }
                // CLAST-style affine-equality predicate: children run
                // only where sum coeffs[p]*vars[p] + offset == 0. The
                // form reads the thread-local var stack only; every
                // accumulation is overflow-checked (Rule 73: no silent
                // wraparound).
                int64_t acc = n.guardOffset;
                for (std::size_t p = 0; p < n.guardCoeffs.size(); ++p) {
                    const int64_t c = n.guardCoeffs[p];
                    if (c == 0) continue;
                    if (p >= vars.size()) {
                        return err(ErrorCode::InvalidGraph,
                                   "walker: guard references an unbound "
                                   "loop var");
                    }
                    int64_t term = 0;
                    if (__builtin_mul_overflow(c, vars[p], &term) ||
                        __builtin_add_overflow(acc, term, &acc)) {
                        return err(ErrorCode::ResourceExhausted,
                                   "walker: guard form overflow");
                    }
                }
                if (acc != 0) break;  // condition false: skip the region
                MLK_TRYV(execChildList(n.children));
                break;
            }
            case KernelOp::Call:
                MLK_TRYV(execGemm(n));
                break;
            case KernelOp::kCount:
                return err(ErrorCode::InvalidGraph,
                           "walker: kCount sentinel");
        }
        return {};
    }

    /// Executes a Compute+Store pair at one loop-nest point.
    [[nodiscard]] Result<void> execPair(const KernelNode& compute,
                                        const KernelNode& store) {
        // Multi-dim computes address memory exclusively through ElemIdx
        // operands; legacy ElemA/ElemB belong to the 1-D fast path.
        for (const KernelExpr& e : compute.exprs) {
            for (const KernelOperand* o : {&e.a, &e.b}) {
                if (o->kind == KernelOperand::Kind::ElemA ||
                    o->kind == KernelOperand::Kind::ElemB) {
                    return err(ErrorCode::InvalidGraph,
                               "walker: legacy element operand in a "
                               "multi-dim compute");
                }
            }
        }
        double* outP = store.bufferOut < kernel.buffers.size() &&
                               kernel.buffers[store.bufferOut].isTemp
                           ? (store.bufferOut < io.temps.size()
                                  ? io.temps[store.bufferOut]
                                  : nullptr)
                           : resolveOutput(kernel, io, store.bufferOut);
        if (outP == nullptr) {
            return err(ErrorCode::InvalidArgument,
                       "walker: output buffer not bound");
        }
        const SinFamily fam =
            compute.family == kInvalidSymbolId
                ? SinFamily::Libm
                : resolveFamily(symbols, compute.family);
        ExprEvaluator ev{compute.exprs,
                         nullptr,
                         nullptr,
                         io.scalars,
                         fam,
                         {},
                         &vars,
                         &kernel,
                         &io};
        KernelOperand target;
        target.kind = KernelOperand::Kind::ElemIdx;
        target.index = static_cast<int64_t>(store.bufferOut);
        target.idxCoeffs = store.outIndexCoeffs;
        target.idxOffset = store.outIndexOffset;
        const int64_t flat = ev.flatIndex(target);
        if (flat < 0) {
            return err(ErrorCode::InvalidGraph,
                       "walker: negative store address");
        }
        // The legacy ElemA/ElemB operand kinds are rejected above, so the
        // per-element index of the 1-D fast path is dead here; 0 keeps the
        // shared evaluator's array accesses trivially in-bounds.
        const double value = ev.run(0);
        double& slot = outP[flat];
        if (store.accum == AccumMode::Add) {
            slot += value;
        } else if (store.accum == AccumMode::Max) {
            // Order-insensitive row-max primitive: NaN never replaces
            // the running value ((NaN > cur) is false); +/-0 ties keep
            // the current slot. Identical select in both emitters'
            // roadmap forms and the reference kernels.
            slot = (value > slot) ? value : slot;
        } else {
            slot = value;
        }
        return {};
    }

    /// Legacy blocked GEMM for Call(MatMul) nodes (shared with the 1-D
    /// executor); ReduceSum/Softmax calls delegate to the shared free
    /// functions.
    [[nodiscard]] Result<void> execGemm(const KernelNode& top) {
        if (top.math == MathOp::ReduceSum) {
            return execReduceSumCall(kernel, io, cancel, top);
        }
        if (top.math == MathOp::Softmax) {
            return execSoftmaxCall(kernel, io, cancel, top);
        }
        if (top.math != MathOp::MatMul) {
            return err(ErrorCode::UnsupportedCapability,
                       "no executor for call target", 121);
        }
        if (top.bufferA >= kernel.buffers.size() ||
            top.bufferB >= kernel.buffers.size() ||
            top.bufferOut >= kernel.buffers.size()) {
            return err(ErrorCode::InvalidGraph,
                       "matmul buffer id out of range");
        }
        const KernelBuffer& ba = kernel.buffers[top.bufferA];
        const KernelBuffer& bb = kernel.buffers[top.bufferB];
        const KernelBuffer& bc = kernel.buffers[top.bufferOut];
        if (ba.dims.size() != 2 || bb.dims.size() != 2 ||
            bc.dims.size() != 2) {
            return err(ErrorCode::InvalidGraph,
                       "matmul requires rank-2 buffer dims");
        }
        const double* a = resolveInput(kernel, io, top.bufferA);
        const double* b = resolveInput(kernel, io, top.bufferB);
        double* c = resolveOutput(kernel, io, top.bufferOut);
        if (a == nullptr || b == nullptr || c == nullptr) {
            return err(ErrorCode::InvalidArgument,
                       "matmul buffers not bound by the caller");
        }
        const int64_t m = ba.dims[0];
        const int64_t k = ba.dims[1];
        const int64_t n = bc.dims[1];
        if (bb.dims[0] != k || bc.dims[0] != m || bc.dims[1] != n) {
            return err(ErrorCode::InvalidGraph,
                       "matmul operand shapes incompatible");
        }
        const SymbolId tmKey = symbols.intern("tile_m");
        const SymbolId tnKey = symbols.intern("tile_n");
        const SymbolId tkKey = symbols.intern("tile_k");
        gemmRowBlock(a, b, c, n, k,
                     scheduleParam(kernel, tmKey,
                                   constants::kDefaultTileM),
                     scheduleParam(kernel, tnKey,
                                   constants::kDefaultTileN),
                     scheduleParam(kernel, tkKey,
                                   constants::kDefaultTileK), 0, m);
        return {};
    }
};

}  // namespace

Result<void> executeKernelOnBuffers(const KernelModule& kernel,
                                    SymbolTable& symbols,
                                    const KernelBufferBindings& io,
                                    CancellationToken* cancel) {
    if (kernel.nodes.empty()) {
        return err(ErrorCode::InvalidGraph, "kernel module has no nodes");
    }
    // Temp scratch materialization (polyhedral synthesis classes): one
    // zero-initialized allocation per isTemp buffer, bufferId-indexed.
    // The storage dies with this function; every element must be
    // written before it is read (the synthesized init statements
    // guarantee that — otherwise the run is defined-but-garbage, never
    // a crash: the flat-index bounds checks still apply).
    KernelBufferBindings exec = io;
    std::vector<std::vector<double>> tempStorage;
    for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
        const KernelBuffer& b = kernel.buffers[bid];
        if (!b.isTemp) continue;
        int64_t n = b.elements;
        if (!b.dims.empty()) {
            n = 1;
            for (const int64_t d : b.dims) {
                if (d <= 0 || n > kKernelTempElementsLimit / d) {
                    return err(ErrorCode::InvalidGraph,
                               "temp buffer element count out of range");
                }
                n *= d;
            }
        }
        if (n <= 0 || n > kKernelTempElementsLimit) {
            return err(ErrorCode::InvalidGraph,
                       "temp buffer element count out of range");
        }
        if (exec.temps.size() < kernel.buffers.size()) {
            exec.temps.resize(kernel.buffers.size(), nullptr);
        }
        tempStorage.emplace_back(static_cast<std::size_t>(n), 0.0);
        exec.temps[bid] = tempStorage.back().data();
    }
    // Multi-dim detection: any ElemIdx operand, affine loop bound,
    // accumulate/max store, buffer-dim loop bound, affine store target,
    // affine guard, or temp buffer routes the whole module through the
    // tree walker (uniform handling of mixed forests).
    bool multiDim = false;
    for (const KernelNode& n : kernel.nodes) {
        if (n.accum != AccumMode::None || !n.beginCoeffs.empty() ||
            !n.endCoeffs.empty() ||
            (n.end == constants::kKernelLoopDynamicBound &&
             n.endBuf != constants::kInvalidId)) {
            multiDim = true;
        }
        if (n.hasAffineStore() ||
            (n.op == KernelOp::Guard && n.hasAffineGuard())) {
            multiDim = true;
        }
        for (const KernelExpr& e : n.exprs) {
            if (e.a.kind == KernelOperand::Kind::ElemIdx ||
                e.b.kind == KernelOperand::Kind::ElemIdx) {
                multiDim = true;
            }
        }
    }
    if (multiDim) {
        MultiDimWalker walker{kernel, symbols, exec, cancel, {}};
        for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
            // Flat-forest roots only (children execute via their parents).
            bool referenced = false;
            for (const KernelNode& n : kernel.nodes) {
                for (const uint32_t c : n.children) {
                    referenced = referenced || c == i;
                }
            }
            if (referenced) continue;
            MLK_TRYV(walker.execNode(i, nullptr));
        }
        if (cancel != nullptr && cancel->cancelled()) {
            return err(ErrorCode::Cancelled,
                       "kernel execution cancelled", 130);
        }
        return {};
    }
    for (const KernelNode& top : kernel.nodes) {
        switch (top.op) {  // Rule 78: exhaustive
            case KernelOp::Loop: {
                int64_t begin = top.begin;
                int64_t end = top.end;
                if (end == constants::kKernelLoopDynamicBound) {
                    end = io.elements;
                }
                if (end < begin) end = begin;
                const std::size_t threads =
                    decideThreads(kernel, symbols, end - begin);
                if (threads <= 1 || top.children.empty()) {
                    for (const uint32_t cid : top.children) {
                        if (cid >= kernel.nodes.size()) {
                            return err(ErrorCode::InvalidGraph,
                                       "kernel child node out of range");
                        }
                        const KernelNode& c = kernel.nodes[cid];
                        if (c.op == KernelOp::Compute) {
                            const KernelNode* store = nullptr;
                            for (const uint32_t sid : top.children) {
                                if (sid < kernel.nodes.size() &&
                                    kernel.nodes[sid].op == KernelOp::Store) {
                                    store = &kernel.nodes[sid];
                                    break;
                                }
                            }
                            runElementwiseRange(c, store != nullptr
                                                       ? *store
                                                       : c,
                                                kernel, symbols, exec, begin,
                                                end, cancel);
                        }
                    }
                } else {
                    // Chunked parallel-for over disjoint ranges; each
                    // chunk polls cancellation (bounded latency).
                    std::vector<std::thread> pool;
                    pool.reserve(threads);
                    const int64_t total = end - begin;
                    const int64_t chunk =
                        (total + static_cast<int64_t>(threads) - 1) /
                        static_cast<int64_t>(threads);
                    for (std::size_t t = 0; t < threads; ++t) {
                        const int64_t b = begin +
                            static_cast<int64_t>(t) * chunk;
                        const int64_t e = std::min(b + chunk, end);
                        if (b >= e) break;
                        pool.emplace_back([&kernel, &symbols, &exec, &top,
                                           b, e, cancel]() {
                            for (const uint32_t cid : top.children) {
                                if (cid >= kernel.nodes.size()) return;
                                const KernelNode& c = kernel.nodes[cid];
                                if (c.op != KernelOp::Compute) continue;
                                const KernelNode* store = nullptr;
                                for (const uint32_t sid : top.children) {
                                    if (sid < kernel.nodes.size() &&
                                        kernel.nodes[sid].op ==
                                            KernelOp::Store) {
                                        store = &kernel.nodes[sid];
                                        break;
                                    }
                                }
                                runElementwiseRange(c,
                                                    store != nullptr
                                                        ? *store
                                                        : c,
                                                    kernel, symbols, exec, b,
                                                    e, cancel);
                            }
                        });
                    }
                    for (auto& th : pool) th.join();
                }
                if (cancel != nullptr && cancel->cancelled()) {
                    return err(ErrorCode::Cancelled,
                               "kernel execution cancelled", 130);
                }
                break;
            }
            case KernelOp::Call: {
                if (top.math == MathOp::ReduceSum) {
                    MLK_TRYV(execReduceSumCall(kernel, exec, cancel, top));
                    break;
                }
                if (top.math == MathOp::Softmax) {
                    MLK_TRYV(execSoftmaxCall(kernel, exec, cancel, top));
                    break;
                }
                if (top.math != MathOp::MatMul) {
                    return err(ErrorCode::UnsupportedCapability,
                               "no executor for call target", 121);
                }
                // GEMM: shapes from buffer dims (A=[M,K], B=[K,N],
                // C=[M,N]); tiles from declarative schedule params.
                if (top.bufferA >= kernel.buffers.size() ||
                    top.bufferB >= kernel.buffers.size() ||
                    top.bufferOut >= kernel.buffers.size()) {
                    return err(ErrorCode::InvalidGraph,
                               "matmul buffer id out of range");
                }
                const KernelBuffer& ba = kernel.buffers[top.bufferA];
                const KernelBuffer& bb = kernel.buffers[top.bufferB];
                const KernelBuffer& bc = kernel.buffers[top.bufferOut];
                if (ba.dims.size() != 2 || bb.dims.size() != 2 ||
                    bc.dims.size() != 2) {
                    return err(ErrorCode::InvalidGraph,
                               "matmul requires rank-2 buffer dims");
                }
                const int64_t m = ba.dims[0];
                const int64_t k = ba.dims[1];
                const int64_t n = bb.dims[1];
                if (bb.dims[0] != k || bc.dims[0] != m ||
                    bc.dims[1] != n) {
                    return err(ErrorCode::InvalidGraph,
                               "matmul operand shapes incompatible");
                }
                const double* a = resolveInput(kernel, exec, top.bufferA);
                const double* b = resolveInput(kernel, exec, top.bufferB);
                double* c = resolveOutput(kernel, exec, top.bufferOut);
                if (a == nullptr || b == nullptr || c == nullptr) {
                    return err(ErrorCode::InvalidArgument,
                               "matmul buffers not bound by the caller");
                }
                if (a == nullptr || b == nullptr || c == nullptr) {
                    return err(ErrorCode::InvalidArgument,
                               "matmul bound to null storage");
                }
                const SymbolId tmKey = symbols.intern("tile_m");
                const SymbolId tnKey = symbols.intern("tile_n");
                const SymbolId tkKey = symbols.intern("tile_k");
                const int64_t tileM = scheduleParam(kernel, tmKey,
                                                    constants::kDefaultTileM);
                const int64_t tileN = scheduleParam(kernel, tnKey,
                                                    constants::kDefaultTileN);
                const int64_t tileK = scheduleParam(kernel, tkKey,
                                                    constants::kDefaultTileK);
                const std::size_t threads =
                    decideThreads(kernel, symbols, m * n);
                if (threads <= 1) {
                    gemmRowBlock(a, b, c, n, k, tileM, tileN, tileK, 0, m);
                } else {
                    std::vector<std::thread> pool;
                    pool.reserve(threads);
                    const int64_t rows = (m + static_cast<int64_t>(threads) -
                                          1) /
                                         static_cast<int64_t>(threads);
                    for (std::size_t t = 0; t < threads; ++t) {
                        const int64_t b0 =
                            static_cast<int64_t>(t) * rows;
                        const int64_t b1 = std::min(b0 + rows, m);
                        if (b0 >= b1) break;
                        pool.emplace_back([=]() {
                            gemmRowBlock(a, b, c, n, k, tileM, tileN,
                                         tileK, b0, b1);
                        });
                    }
                    for (auto& th : pool) th.join();
                }
                if (cancel != nullptr && cancel->cancelled()) {
                    return err(ErrorCode::Cancelled,
                               "kernel execution cancelled", 130);
                }
                break;
            }
            case KernelOp::Compute:
            case KernelOp::Load:
            case KernelOp::Store:
            case KernelOp::AllocBuffer:
            case KernelOp::CopyBuffer:
            case KernelOp::Barrier:
            case KernelOp::Trace:
                // Non-top-level ops are loop-body concerns; standalone
                // occurrences are ignored by this executor on purpose.
                break;
            case KernelOp::Guard:
                // Guards belong to the guarded execution path
                // (ExecutionEngine), not the raw buffer ABI.
                break;
            case KernelOp::kCount:
                return err(ErrorCode::InvalidGraph,
                           "kernel module contains kCount sentinel");
        }
    }
    return {};
}

}  // namespace mlk
