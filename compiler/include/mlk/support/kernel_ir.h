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
