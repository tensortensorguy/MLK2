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

/// Elementwise math performed by a Compute node (subset of MathOp that has
/// a scalar realization in the CPU backend).
[[nodiscard]] bool isScalarRealizable(MathOp op) noexcept;

/// Operand of a KernelExpr: a compile-time constant, the current element
/// of input buffer A/B, a runtime scalar parameter, or the result of a
/// prior expression in the same Compute chain (SSA over temps).
struct KernelOperand {
    enum class Kind : uint8_t {
        Const = 0,   // value = constValue
        ElemA,       // bufferA[i]  (loop element)
        ElemB,       // bufferB[i]  (loop element)
        ScalarParam, // runtime scalar, index = scalarIndex
        Temp,        // exprs[tempIndex] result (tempIndex < self)
    };
    Kind kind{Kind::Const};
    int64_t index{0};        // Temp: expr index; ScalarParam: param index
    double constValue{0.0};  // Kind::Const payload

    [[nodiscard]] bool operator==(const KernelOperand& o) const noexcept {
        return kind == o.kind && index == o.index &&
               constValue == o.constValue;
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
    /// Buffer/param references (indices into KernelModule arrays).
    uint32_t bufferA{constants::kInvalidId};
    uint32_t bufferB{constants::kInvalidId};
    uint32_t bufferOut{constants::kInvalidId};
    uint32_t guardId{constants::kInvalidId};  // GraphState ref (Rule 5)
    /// Implementation family for math functions (libm / poly7; Rule 34:
    /// carries the verified accuracy contract reference).
    SymbolId family{kInvalidSymbolId};
    /// Fused per-element expression chain (Compute). Empty => legacy single
    /// op semantics (compute.math over ElemA/ElemB).
    SmallVector<KernelExpr, 8> exprs{};
    SmallVector<uint32_t, 4> children{};      // loop body / region nodes

    [[nodiscard]] bool isLoop() const { return op == KernelOp::Loop; }
};

struct KernelBuffer {
    SymbolId name{kInvalidSymbolId};
    Dtype dtype{Dtype::F32};
    bool isInput{false};
    bool isOutput{false};
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
