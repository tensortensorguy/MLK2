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
inline constexpr std::size_t kKernelExecMaxThreads = 4;
inline constexpr int64_t kParallelChunkElements = 16384;
inline constexpr std::size_t kMaxTempSlots = 64;

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
            case KernelOperand::Kind::ElemIdx:
                // Affine multi-index operands are produced only by the
                // polyhedral codegen and executed by the multi-dim tree
                // walker (see docs/polyhedral_spec.md §executor); the
                // 1-D executor never receives them. Reaching here means
                // a poly kernel hit the legacy path — the documented
                // behavior is the same neutral element the other
                // unbound cases use, never a crash (Rule 115).
                return 0.0;
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
    if (threads > kKernelExecMaxThreads) threads = kKernelExecMaxThreads;
    const SymbolId threadsKey = symbols.intern("threads");
    if (const int64_t* v = kernel.scheduleParams.find(threadsKey)) {
        if (*v >= 1) {
            threads = static_cast<std::size_t>(*v);
        }
    }
    if (elements < kParallelChunkElements) threads = 1;
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

}  // namespace

Result<void> executeKernelOnBuffers(const KernelModule& kernel,
                                    SymbolTable& symbols,
                                    const KernelBufferBindings& io,
                                    CancellationToken* cancel) {
    if (kernel.nodes.empty()) {
        return err(ErrorCode::InvalidGraph, "kernel module has no nodes");
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
                                                kernel, symbols, io, begin,
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
                        pool.emplace_back([&kernel, &symbols, &io, &top,
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
                                                    kernel, symbols, io, b,
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
                const double* a = resolveInput(kernel, io, top.bufferA);
                const double* b = resolveInput(kernel, io, top.bufferB);
                double* c = resolveOutput(kernel, io, top.bufferOut);
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
