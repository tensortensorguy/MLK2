// Cost model implementation: table-driven costs, runtime hardware probe.
#include "mlk/cost/cost_model.h"

#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace mlk {

HardwareInfo HardwareInfo::detect() {
    HardwareInfo hw;
    hw.cores = static_cast<uint32_t>(std::thread::hardware_concurrency());
    if (hw.cores == 0) hw.cores = 1;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        hw.simdWidthF32 = 16;
        hw.peakFlopsPerCycleF32 = 64.0;
        hw.peakFlopsPerCycleF64 = 32.0;
    } else if (__builtin_cpu_supports("avx2")) {
        hw.simdWidthF32 = 8;
        hw.peakFlopsPerCycleF32 = 32.0;
        hw.peakFlopsPerCycleF64 = 16.0;
    } else if (__builtin_cpu_supports("sse2")) {
        hw.simdWidthF32 = 4;
        hw.peakFlopsPerCycleF32 = 8.0;
        hw.peakFlopsPerCycleF64 = 4.0;
    }
    hw.hasFMA = __builtin_cpu_supports("fma") != 0;
#endif
    return hw;
}

HashValue HardwareInfo::fingerprint() const noexcept {
    HashValue h = hashU64(cacheLineBytes);
    h = hashCombine(h, hashU64(simdWidthF32));
    h = hashCombine(h, hashU64(cores));
    h = hashCombine(h, hashF64(peakFlopsPerCycleF32));
    h = hashCombine(h, hashF64(memoryBandwidthGiBs));
    h = hashCombine(h, hashU64(hasFMA));
    return h;
}

CostEstimate CostModel::graphCost(const MathGraph& graph,
                                  const HardwareInfo& hw) {
    CostEstimate total;
    BasicCostModel basic;
    for (const NodeId nid : graph.topoOrder()) {
        const CostEstimate c = basic.nodeCost(graph, nid, hw);
        total.flops += c.flops;
        total.bytesMoved += c.bytesMoved;
        total.instructions += c.instructions;
        total.latencyNs += c.latencyNs;
    }
    return total;
}

CostEstimate BasicCostModel::nodeCost(const MathGraph& graph, NodeId nodeId,
                                      const HardwareInfo& hw) {
    const Node& n = graph.node(nodeId);
    CostEstimate c;
    // Shape-derived element counts (Rule 55: cost from shape/hardware facts,
    // not vibes).
    double elements = 1.0;
    const Value& result = graph.value(n.results[0]);
    if (result.type.shape) {
        const auto num = result.type.shape->numel();
        elements = num.has_value() ? static_cast<double>(*num) : elements;
    } else if (result.type.tensor) {
        const auto num = result.type.tensor->shape.numel();
        elements = num.has_value() ? static_cast<double>(*num) : elements;
    }
    const int elemBytes = result.type.dtype != Dtype::None
                              ? dtypeBytes(result.type.dtype)
                              : 4;

    switch (n.op) {  // Rule 78: exhaustive; named constant kCount is the
                     // sentinel, per-element costs are table values
        case MathOp::Add: case MathOp::Sub: case MathOp::Mul:
        case MathOp::Neg:
            c.flops = elements;
            c.instructions = elements;
            break;
        case MathOp::Div:
            c.flops = elements;
            c.instructions = elements;
            c.latencyNs += elements * 1.0;  // div latency dominates
            break;
        case MathOp::Pow:
            c.flops = elements * 8.0;
            c.instructions = elements * 8.0;
            break;
        case MathOp::Exp: case MathOp::Log: case MathOp::Sin:
        case MathOp::Cos: case MathOp::Tan: case MathOp::Tanh:
        case MathOp::Sqrt: case MathOp::Rsqrt: case MathOp::Erf:
            c.flops = elements;
            c.instructions = elements;
            c.latencyNs += elements * 8.0;  // libm-class latency per call
            break;
        case MathOp::Gelu:
            c.flops = elements * 3.0;
            c.instructions = elements * 4.0;
            break;
        case MathOp::MatMul: {
            // 2*M*N*K flops for the contraction.
            if (result.type.tensor) {
                const auto& s = result.type.tensor->shape;
                if (s.rank() == 2 && s.isStatic()) {
                    const MathType& ta = graph.value(n.inputs[0]).type;
                    const int64_t k = ta.tensor ? ta.tensor->shape.dim(1) : 0;
                    c.flops = 2.0 * static_cast<double>(s.dim(0)) *
                              static_cast<double>(s.dim(1)) *
                              static_cast<double>(k);
                    c.instructions = c.flops;
                }
            }
            break;
        }
        case MathOp::Dot: case MathOp::ReduceSum: case MathOp::ReduceMax:
        case MathOp::ReduceMean:
            c.flops = elements;
            c.instructions = elements;
            break;
        case MathOp::Transpose: case MathOp::Reshape:
        case MathOp::Broadcast: case MathOp::LayoutTransform:
            c.instructions = elements;
            c.bytesMoved = elements * elemBytes;
            break;
        case MathOp::Softmax:
            c.flops = elements * 5.0;
            c.instructions = elements * 6.0;
            c.bytesMoved = elements * elemBytes * 2.0;
            break;
        case MathOp::Einsum: case MathOp::Conv:
            c.flops = elements * 8.0;
            c.instructions = c.flops;
            break;
        case MathOp::Derivative: case MathOp::Integral:
        case MathOp::Gradient: case MathOp::Limit:
        case MathOp::Solve: case MathOp::Apply: case MathOp::Lambda:
            // Symbolic-level ops have no direct cost; their realization
            // decides (strategy layer).
            break;
        case MathOp::IntToFloat: case MathOp::FloatToInt:
        case MathOp::RealToComplex: case MathOp::ScalarToTensor:
        case MathOp::TensorToScalar: case MathOp::Reinterpret:
        case MathOp::BitCast: case MathOp::Box: case MathOp::Unbox:
        case MathOp::NativeToMathRef: case MathOp::MathToNativeRef:
            c.instructions = elements;
            break;
        case MathOp::kCount:
            break;
    }
    if (c.bytesMoved == 0.0 && elements > 1.0) {
        c.bytesMoved = elements * elemBytes;  // one read+write pass approx
    }
    const double peakFlopsPerCycle =
        elemBytes == 8 ? hw.peakFlopsPerCycleF64 : hw.peakFlopsPerCycleF32;
    const double cycles = peakFlopsPerCycle > 0.0
                              ? c.flops / peakFlopsPerCycle
                              : 0.0;
    c.latencyNs += cycles / hw.clockGHz;
    return c;
}

double rooflineLowerBoundNs(const CostEstimate& c, const HardwareInfo& hw) {
    const double peakFlopsPerSec = hw.cores * hw.peakFlopsPerCycleF32 *
                                   hw.clockGHz * 1e9;
    const double bwBytesPerSec =
        hw.memoryBandwidthGiBs * 1024.0 * 1024.0 * 1024.0;
    double ns = 0.0;
    if (peakFlopsPerSec > 0.0) ns = c.flops / peakFlopsPerSec * 1e9;
    if (bwBytesPerSec > 0.0) {
        const double memNs = c.bytesMoved / bwBytesPerSec * 1e9;
        if (memNs > ns) ns = memNs;
    }
    return ns;
}

double rooflineCandidateLowerBoundNs(const CostEstimate& c,
                                     const HardwareInfo& hw, int64_t threads,
                                     int64_t vectorWidth) {
    // Resource fractions (Rule 27: named, bounded, defensible). A thread
    // count above the machine's core count is clamped to 1.0 — extra
    // threads cannot create cores. An unset knob (<= 0) keeps the
    // machine-wide peak.
    double coreFrac = 1.0;
    if (threads > 0 && hw.cores > 0) {
        const double t = static_cast<double>(threads);
        const double cores = static_cast<double>(hw.cores);
        coreFrac = t < cores ? t / cores : 1.0;
    }
    double vecFrac = 1.0;
    if (vectorWidth > 0 && hw.simdWidthF32 > 0) {
        const double v = static_cast<double>(vectorWidth);
        const double simdw = static_cast<double>(hw.simdWidthF32);
        vecFrac = v < simdw ? v / simdw : 1.0;
    }
    const double peakFlopsPerSec = hw.cores * hw.peakFlopsPerCycleF32 *
                                   hw.clockGHz * 1e9 * coreFrac * vecFrac;
    const double bwBytesPerSec =
        hw.memoryBandwidthGiBs * 1024.0 * 1024.0 * 1024.0 * coreFrac;
    double ns = 0.0;
    if (peakFlopsPerSec > 0.0) ns = c.flops / peakFlopsPerSec * 1e9;
    if (bwBytesPerSec > 0.0) {
        const double memNs = c.bytesMoved / bwBytesPerSec * 1e9;
        if (memNs > ns) ns = memNs;
    }
    return ns;
}

}  // namespace mlk
