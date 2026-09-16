// MLK+ KernelIR (lower.to_kernel_ir target; spec §12: "Do not emit machine
// code directly from the mathematical graph. Put a kernel IR between math
// and backend").
//
// KernelModule is explicit about: loops, memories (buffers), computes,
// barriers, kernel boundaries, parallel axes. It is serializable and
// hashable (Rule 24). Schedule decisions (tile/vectorize/parallelize) attach
// as named parameters consumed by executors and emitters.
#pragma once

#include "mlk/core/constants.h"
#include "mlk/core/hash_map.h"
#include "mlk/type/domain.h"
#include "mlk/core/hash.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_op.h"
#include "mlk/support/json.h"

namespace mlk {

enum class KernelOp : uint8_t {
    Loop = 0,       // for (i = begin; i < end; i += step)
    Compute,        // scalar math op over buffer/param elements
    Load,           // buffer element -> temp
    Store,          // temp -> buffer element
    AllocBuffer,    // materialized buffer
    CopyBuffer,
    Barrier,
    Trace,          // tracing hook (Rule 96: must remain correct)
    Guard,          // speculative guard w/ fallback (Rule 5)
    Call,           // approved runtime entrypoint (Rule 121)
    kCount,
};

const char* kernelOpName(KernelOp op) noexcept;

/// Store update mode (polyhedral extension). `Add` is the reduction
/// primitive (out += value, Rule 90 accumulation order); `Max` is the
/// order-insensitive row-max primitive used by stable softmax synthesis
/// (out = (value > out) ? value : out — NaN never replaces the running
/// value; see docs/polyhedral_spec.md §synthesis).
enum class AccumMode : uint8_t {
    None = 0,
    Add = 1,
    Max = 2,
};

/// Elementwise math performed by a Compute node (subset of MathOp that has
/// a scalar realization in the CPU backend).
[[nodiscard]] bool isScalarRealizable(MathOp op) noexcept;

/// Operand of a KernelExpr: a compile-time constant, the current element
/// of input buffer A/B, a runtime scalar parameter, the result of a prior
/// expression in the same Compute chain (SSA over temps), or an AFFINE
/// multi-index element of any buffer (polyhedral codegen; see
/// docs/polyhedral_spec.md §kernel-ir-extension).
struct KernelOperand {
    enum class Kind : uint8_t {
        Const = 0,   // value = constValue
        ElemA,       // bufferA[i]  (loop element)
        ElemB,       // bufferB[i]  (loop element)
        ScalarParam, // runtime scalar, index = scalarIndex
        Temp,        // exprs[tempIndex] result (tempIndex < self)
        ElemIdx,     // buffer[index]: flat = sum idxCoeffs[d]*var[d] +
                     // idxOffset (row-major dense element address)
    };
    Kind kind{Kind::Const};
    int64_t index{0};        // Temp: expr index; ScalarParam: param index;
                             // ElemIdx: buffer id
    double constValue{0.0};  // Kind::Const payload
    /// Kind::ElemIdx only: coefficient per enclosing loop var (stack order,
    /// outermost first) and constant offset. Strides are baked by the
    /// emitter from the buffer's row-major dims.
    SmallVector<int64_t, 4> idxCoeffs{};
    int64_t idxOffset{0};

    [[nodiscard]] bool operator==(const KernelOperand& o) const noexcept {
        return kind == o.kind && index == o.index &&
               constValue == o.constValue && idxCoeffs == o.idxCoeffs &&
               idxOffset == o.idxOffset;
    }
};

/// One scalar math operation in a fused per-element Compute chain.
/// A Compute node evaluates exprs[] in order; unary ops use only `a`.
/// The last expression's value is the Compute result (Rule 24: the chain
/// is part of the serializable, hashable KernelModule).
struct KernelExpr {
    MathOp op{MathOp::Add};
    KernelOperand a{};
    KernelOperand b{};  // ignored for unary ops

    [[nodiscard]] bool operator==(const KernelExpr& o) const noexcept {
        return op == o.op && a == o.a && b == o.b;
    }
};

struct KernelNode {
    KernelOp op{KernelOp::Loop};
    MathOp math{MathOp::Add};          // for Compute
    SymbolId var{kInvalidSymbolId};    // loop induction var name
    int64_t begin{0};
    int64_t end{constants::kKernelLoopDynamicBound};
    int64_t step{1};
    /// Polyhedral extension: affine bounds over the ENCLOSING loop-var
    /// stack (outermost first). When beginCoeffs is non-empty:
    ///   begin        = sum beginCoeffs[d]*var[d] + beginOffset
    ///   endInclusive = sum endCoeffs[d]*var[d]   + endOffset
    /// (the executor iterates var from begin to endInclusive inclusive;
    /// see docs/polyhedral_spec.md). Empty => the constant begin/end
    /// above (backward compatible).
    SmallVector<int64_t, 4> beginCoeffs{};
    int64_t beginOffset{0};
    SmallVector<int64_t, 4> endCoeffs{};
    int64_t endOffset{0};
    /// Polyhedral extension: when end == kKernelLoopDynamicBound and
    /// endBuf is valid, the bound is buffers[endBuf].dims[endDim] resolved
    /// at execution (multi-dim nests from poly.synth/poly.codegen).
    uint32_t endBuf{constants::kInvalidId};
    int32_t endDim{-1};
    /// Buffer/param references (indices into KernelModule arrays).
    uint32_t bufferA{constants::kInvalidId};
    uint32_t bufferB{constants::kInvalidId};
    uint32_t bufferOut{constants::kInvalidId};
    /// Polyhedral extension (Store): affine flat target
    /// out[sum outIndexCoeffs[d]*var[d] + outIndexOffset]. Empty coeffs =>
    /// legacy 1-D semantics (out[i]).
    SmallVector<int64_t, 4> outIndexCoeffs{};
    int64_t outIndexOffset{0};
    /// Polyhedral extension (Store): update mode for the affine target
    /// (AccumMode::Add = out[flat] += value, the reduction primitive,
    /// Rule 90: same accumulation order as the reference; AccumMode::Max
    /// = out[flat] = (value > out[flat]) ? value : out[flat], the
    /// order-insensitive row-max primitive).
    AccumMode accum{AccumMode::None};
    /// Polyhedral extension (Loop): the scheduler proved every
    /// dependence distance identically zero at this level — instances
    /// with different induction values are independent, so the executor
    /// may run disjoint index chunks on threads (deterministic: slabs
    /// read/write disjoint locations at this level).
    bool parallel{false};
    /// Polyhedral extension (Loop): innermost parallel level whose active
    /// statements access memory with element stride 0/1 along this dim
    /// (SIMD-able; advisory — the executor decides, Rule 12).
    bool vectorHint{false};
    uint32_t guardId{constants::kInvalidId};  // GraphState ref (Rule 5)
    /// Polyhedral extension (Guard): affine equality condition over the
    /// ENCLOSING loop-var stack (outermost first). When guardCoeffs is
    /// non-empty the node is a CLAST-style predicate: its children run
    /// only where
    ///   sum guardCoeffs[p]*var[p] + guardOffset == 0
    /// (poly.codegen guarded statement re-entry; see docs/polyhedral_spec.md
    /// §codegen). Empty => legacy speculative guard semantics (Rule 5).
    SmallVector<int64_t, 4> guardCoeffs{};
    int64_t guardOffset{0};
    /// Implementation family for math functions (libm / poly7; Rule 34:
    /// carries the verified accuracy contract reference).
    SymbolId family{kInvalidSymbolId};
    /// Fused per-element expression chain (Compute). Empty => legacy single
    /// op semantics (compute.math over ElemA/ElemB).
    SmallVector<KernelExpr, 8> exprs{};
    SmallVector<uint32_t, 4> children{};      // loop body / region nodes

    [[nodiscard]] bool isLoop() const { return op == KernelOp::Loop; }
    /// True when the Store target is the polyhedral affine form.
    [[nodiscard]] bool hasAffineStore() const noexcept {
        return !outIndexCoeffs.empty();
    }
    /// True when the Guard carries the polyhedral affine-equality form.
    [[nodiscard]] bool hasAffineGuard() const noexcept {
        return !guardCoeffs.empty();
    }
};

struct KernelBuffer {
    SymbolId name{kInvalidSymbolId};
    Dtype dtype{Dtype::F32};
    bool isInput{false};
    bool isOutput{false};
    /// Polyhedral extension: executor-allocated scratch (neither bound
    /// as input nor output). Synthesis classes that materialize
    /// intermediate values (softmax rowmax/exp/sum temps) declare their
    /// temporaries this way; the buffer executors zero-initialize the
    /// storage and every element must be written before it is read
    /// (the init statements of the synthesized nests guarantee that).
    bool isTemp{false};
    int64_t elements{constants::kKernelLoopDynamicBound};
    uint32_t alignment{static_cast<uint32_t>(
        constants::kDefaultAlignmentBytes)};
    /// Logical dimensions (row-major; empty for 1-D/dynamic buffers).
    /// Required by structured kernels (GEMM: A=[M,K], B=[K,N], C=[M,N]).
    SmallVector<int64_t, 4> dims{};
};

struct KernelModule {
    SmallVector<KernelBuffer, 4> buffers{};
    SmallVector<KernelNode, 8> nodes{};   // nodes[0] is the root loop/region
    SymbolId name{kInvalidSymbolId};
    /// Schedule parameters (tile/vector/unroll/threads; Rule 54 declarative).
    OpenHashMap<SymbolId, int64_t> scheduleParams{};

    [[nodiscard]] uint32_t addBuffer(KernelBuffer b) {
        b.dtype = b.dtype;
        const auto id = static_cast<uint32_t>(buffers.size());
        buffers.push_back(b);
        return id;
    }
    [[nodiscard]] uint32_t addNode(KernelNode n) {
        const auto id = static_cast<uint32_t>(nodes.size());
        nodes.push_back(n);
        return id;
    }

    [[nodiscard]] HashValue hash() const noexcept;
    [[nodiscard]] json::Value toJson(SymbolTable& symbols) const;
};

/// Serializes a schedule param (typed accessors used by executors).
[[nodiscard]] inline int64_t scheduleParam(const KernelModule& m,
                                           SymbolId name, int64_t fallback) {
    const int64_t* v = m.scheduleParams.find(name);
    return v != nullptr ? *v : fallback;
}

}  // namespace mlk
