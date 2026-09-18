// CUDA source emitter implementation (see header for the artifact
// contract). Semantics mirror runtime/src/execution/kernel_buffers.cpp
// EXACTLY — the MultiDimWalker and the 1-D executor — so a differential
// test between the buffer walker and the compiled artifact is
// bit-exact-by-construction for the device-exact op set (the header
// declares the ULP-bounded boundary for device-libm transcendentals;
// the driver passes --fmad=false so no mul/add ever contracts, Rules
// 33/90). Tests: tests/unit/unit_poly.cpp cuda_emission_*.
#include "mlk/backend/cuda_emitter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mlk/backend/cpp_emitter.h"
#include "mlk/core/constants.h"

namespace mlk {

namespace {

/// Same temp-slot cap as the executor and the other emitters (Rule 27).
inline constexpr std::size_t kCudaMaxTempSlots = 64;
/// One __global__ per forest root; beyond this the emission is refused
/// honestly (Rule 27: named cap, structured rejection — never a silent
/// truncation).
inline constexpr std::size_t kCudaMaxDeviceKernels = 64;
/// Launch geometry cap (CUDA compute-capability bound for x-dimension
/// block sizes across all supported architectures).
inline constexpr int64_t kCudaMaxThreadsPerBlock = 1024;

[[nodiscard]] bool isUnaryOp(MathOp op) noexcept {
    return op == MathOp::Neg || op == MathOp::Exp || op == MathOp::Log ||
           op == MathOp::Sin || op == MathOp::Cos || op == MathOp::Tan ||
           op == MathOp::Tanh || op == MathOp::Sqrt ||
           op == MathOp::Rsqrt || op == MathOp::Erf || op == MathOp::Gelu;
}

[[nodiscard]] std::string i64(const int64_t v) { return std::to_string(v); }

/// Exact double literal for DEVICE code: NaN/inf use bit-exact intrinsic
/// forms (device libm macros are not guaranteed identical; the bit
/// pattern is the contract). Finite values round-trip through %.17g with
/// a forced float form.
[[nodiscard]] std::string f64(const double d) {
    if (std::isnan(d)) {
        return "__longlong_as_double(0x7ff8000000000000ULL)";
    }
    if (std::isinf(d)) {
        return d > 0.0 ? "__longlong_as_double(0x7ff0000000000000ULL)"
                       : "__longlong_as_double(0xfff0000000000000ULL)";
    }
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.17g", d);
    std::string s = buf;
    if (s.find_first_of(".eEnN") == std::string::npos) s += ".0";
    return s;
}

/// C identifier sanitizer (same discipline as the C++ emitter).
[[nodiscard]] std::string sanitize(std::string raw) {
    if (raw.empty()) return "v";
    for (char& c : raw) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_';
        if (!ok) c = '_';
    }
    if (raw[0] >= '0' && raw[0] <= '9') raw = "_" + raw;
    static const char* kReserved[] = {"int",     "double", "float",  "for",
                                      "if",      "else",   "return", "void",
                                      "const",   "auto",   "while",  "break",
                                      "continue"};
    for (const char* r : kReserved) {
        if (raw == r) return "mlk_" + raw;
    }
    return raw;
}

struct CudaEmitter {
    const KernelModule& km;
    SymbolTable& symbols;
    bool multiDim{false};
    bool needsPoly7{false};
    bool transcendental{false};  // any device-libm op -> ULP-bounded policy
    std::size_t parallelMarks{0};
    std::size_t nestedParallelSerial{0};  // marks below the collapse point
    std::size_t collapsedRoots{0};
    std::size_t serialRoots{0};
    std::size_t simdLoops{0};  // vectorHint recorded, advisory on device
    std::string devBody{};     // __global__ kernels
    std::string hostBody{};    // extern "C" mlk_kernel wrapper
    int depth{0};
    SmallVector<std::string, 8> varNames{};
    std::vector<std::string> used{};
    std::vector<std::string> ptrName{};
    std::vector<std::string> dimsName{};
    std::vector<bool> stored{};
    std::vector<bool> bindable{};

    [[nodiscard]] std::string uniqueName(std::string base) {
        base = sanitize(std::move(base));
        std::string candidate = base;
        std::size_t suffix = used.size();
        while (true) {
            bool clash = false;
            for (const std::string& u : used) clash = clash || u == candidate;
            if (!clash) break;
            candidate = base + "_" + std::to_string(suffix++);
        }
        used.push_back(candidate);
        return candidate;
    }

    [[nodiscard]] std::string indent() const {
        return std::string(static_cast<std::size_t>(depth) * 4, ' ');
    }

    /// Affine form over an EXPLICIT name stack (walker flatIndex
    /// contract: sum coeffs[p]*var[p] + offset; zero coefficients never
    /// bind a var). Explicit parentheses everywhere; no re-association —
    /// integer address arithmetic, exact. The node walk passes the live
    /// varNames; the padded-collapse analysis passes the prefix names
    /// (which do not exist as text yet at analysis time).
    [[nodiscard]] Result<std::string> affineOver(
        const SmallVector<int64_t, 4>& coeffs, const int64_t offset,
        const SmallVector<std::string, 8>& names) const {
        std::string s;
        bool any = false;
        for (std::size_t p = 0; p < coeffs.size(); ++p) {
            const int64_t c = coeffs[p];
            if (c == 0) continue;
            if (p >= names.size()) {
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: affine form references an "
                           "unbound loop var");
            }
            std::string term;
            if (c == 1) {
                term = names[p];
            } else if (c == -1) {
                term = "-" + names[p];
            } else {
                term = i64(c) + "*" + names[p];
            }
            if (any) {
                s += term[0] == '-' ? " - " + term.substr(1) : " + " + term;
            } else {
                s = term;
                any = true;
            }
        }
        if (offset != 0 || !any) {
            if (!any) {
                s = i64(offset);
            } else if (offset > 0) {
                s += " + " + i64(offset);
            } else {
                s += " - " + i64(-offset);
            }
        }
        return "(" + s + ")";
    }

    [[nodiscard]] Result<std::string> affine(
        const SmallVector<int64_t, 4>& coeffs, const int64_t offset) const {
        return affineOver(coeffs, offset, varNames);
    }

    /// Validated buffer pointer name (walker resolveInput/resolveOutput
    /// contract: input | output | temp are bindable).
    [[nodiscard]] Result<std::string> bufRef(const uint32_t bid) const {
        if (bid >= km.buffers.size()) {
            return err(ErrorCode::InvalidGraph,
                       "cuda emitter: buffer id out of range");
        }
        const KernelBuffer& b = km.buffers[bid];
        if (!b.isInput && !b.isOutput && !b.isTemp) {
            return err(ErrorCode::InvalidGraph,
                       "cuda emitter: reference to an unbindable buffer "
                       "(neither input, output, nor temp)");
        }
        return ptrName[bid];
    }

    /// One scalar operand expression (walker evalOne contract).
    [[nodiscard]] Result<std::string> operand(const KernelOperand& o,
                                              const KernelNode& compute) {
        switch (o.kind) {  // Rule 78: exhaustive
            case KernelOperand::Kind::Const:
                return f64(o.constValue);
            case KernelOperand::Kind::ElemA:
            case KernelOperand::Kind::ElemB: {
                if (multiDim) {
                    return err(ErrorCode::InvalidGraph,
                               "cuda emitter: legacy element operand in a "
                               "multi-dim compute");
                }
                if (varNames.empty()) {
                    return err(ErrorCode::InvalidGraph,
                               "cuda emitter: legacy operand outside a "
                               "loop");
                }
                const uint32_t bid = o.kind == KernelOperand::Kind::ElemA
                                         ? compute.bufferA
                                         : compute.bufferB;
                auto ptr = bufRef(bid);
                if (!ptr.has_value()) {
                    return std::unexpected<Error>(ptr.error());
                }
                return *ptr + "[" + varNames.back() + "]";
            }
            case KernelOperand::Kind::ScalarParam: {
                return "((" + i64(o.index) + " < mlk_n_scalars) ? "
                            "mlk_scalars[" + i64(o.index) + "] : 0.0)";
            }
            case KernelOperand::Kind::Temp: {
                if (o.index < 0 ||
                    static_cast<std::size_t>(o.index) >=
                        kCudaMaxTempSlots) {
                    return err(ErrorCode::InvalidGraph,
                               "cuda emitter: temp index out of range");
                }
                return "mlk_t" + i64(o.index);
            }
            case KernelOperand::Kind::ElemIdx: {
                auto ptr = bufRef(static_cast<uint32_t>(o.index));
                if (!ptr.has_value()) {
                    return std::unexpected<Error>(ptr.error());
                }
                auto flat = affine(o.idxCoeffs, o.idxOffset);
                if (!flat.has_value()) {
                    return std::unexpected<Error>(flat.error());
                }
                return *ptr + "[" + *flat + "]";
            }
        }
        return err(ErrorCode::InvalidGraph,
                   "cuda emitter: unknown operand kind");
    }

    /// One scalar op application (walker apply() contract verbatim —
    /// singularity-as-value special cases and the Pow(2,2) fast form;
    /// Rule 93). Device-exact ops (IEEE mul/add/div/sqrt, floor/fmod,
    /// the poly7 snapshot) keep the bit-exact claim; device-libm forms
    /// flip the module policy to ULP-bounded (declared, never silent).
    [[nodiscard]] Result<std::string> apply(const KernelExpr& e,
                                            const KernelNode& compute) {
        auto a = operand(e.a, compute);
        if (!a.has_value()) return std::unexpected<Error>(a.error());
        const bool unary = isUnaryOp(e.op);
        std::string b;
        if (!unary) {
            auto bv = operand(e.b, compute);
            if (!bv.has_value()) return std::unexpected<Error>(bv.error());
            b = *bv;
        }
        const bool poly7Sin =
            e.op == MathOp::Sin &&
            compute.family != kInvalidSymbolId &&
            symbols.text(compute.family) == "poly7";
        if (poly7Sin) needsPoly7 = true;
        switch (e.op) {  // Rule 78: exhaustive over scalar-realizable ops
            case MathOp::Add: return "(" + *a + ") + (" + b + ")";
            case MathOp::Sub: return "(" + *a + ") - (" + b + ")";
            case MathOp::Mul: return "(" + *a + ") * (" + b + ")";
            case MathOp::Div:
                // IEEE division: correctly rounded (nvcc default
                // -prec-div=true); the guard mirrors the walker.
                return "((" + b + ") != 0.0 ? (" + *a + ") / (" + b +
                       ") : 0.0)";
            case MathOp::Neg: return "-(" + *a + ")";
            case MathOp::Pow:
                // The non-fast form is device libm pow — ULP-bounded.
                transcendental = true;
                return "(((" + *a + ") == 2.0 && (" + b + ") == 2.0) ? (" +
                       *a + ") * (" + *a + ") : pow((" + *a + "), (" +
                       b + ")))";
            case MathOp::Exp:
                transcendental = true;
                return "exp(" + *a + ")";
            case MathOp::Log:
                transcendental = true;
                return "((" + *a + ") > 0.0 ? log(" + *a + ") : 0.0)";
            case MathOp::Sin:
                if (poly7Sin) break;  // device-exact snapshot below
                transcendental = true;
                return "sin(" + *a + ")";
            case MathOp::Cos:
                transcendental = true;
                return "cos(" + *a + ")";
            case MathOp::Tan:
                transcendental = true;
                return "tan(" + *a + ")";
            case MathOp::Tanh:
                transcendental = true;
                return "tanh(" + *a + ")";
            case MathOp::Sqrt:
                // IEEE correctly rounded on the device — device-exact.
                return "((" + *a + ") >= 0.0 ? sqrt(" + *a + ") : 0.0)";
            case MathOp::Rsqrt:
                return "((" + *a + ") > 0.0 ? 1.0 / sqrt(" + *a +
                       ") : 0.0)";
            case MathOp::Erf:
                transcendental = true;
                return "erf(" + *a + ")";
            case MathOp::Gelu:
                transcendental = true;
                return "0.5 * (" + *a +
                       ") * (1.0 + tanh(0.7978845608028654 * ((" + *a +
                       ") + 0.044715 * (" + *a + ") * (" + *a + ") * (" +
                       *a + "))))";
            default:
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: non-realizable math op");
        }
        return "mlk_poly7_sin(" + *a + ")";
    }

    /// Emits the multi-dim store to a validated affine flat target with
    /// the negative-flat structural check (the walker rejects negative
    /// store flats with InvalidGraph; the artifact reports the same
    /// violation through the status slot — the assembly artifact's ABI
    /// code 1). AccumMode::Max emits the walker's EXACT select.
    [[nodiscard]] Result<void> emitAffineStore(const KernelNode& store,
                                               const std::string& value) {
        auto ptr = bufRef(store.bufferOut);
        if (!ptr.has_value()) {
            return std::unexpected<Error>(ptr.error());
        }
        MLK_TRY_VAR(flat, affine(store.outIndexCoeffs,
                                 store.outIndexOffset));
        body() += indent() + "{ const int64_t mlk_flat = " + flat + ";\n";
        ++depth;
        body() += indent() +
                  "if (mlk_flat < 0) { *mlk_status = 1; return; }\n";
        const std::string target = *ptr + "[mlk_flat]";
        if (store.accum == AccumMode::Max) {
            // Order-insensitive row-max — the EXACT walker select
            // (execPair): a NaN value never replaces the running slot
            // ((NaN > cur) is false), +/-0 ties keep the current bits.
            body() += indent() + target + " = (" + value + " > " + target +
                      ") ? " + value + " : " + target + ";\n";
        } else {
            body() += indent() + target +
                      (store.accum == AccumMode::Add ? " += " : " = ") +
                      value + ";\n";
        }
        --depth;
        body() += indent() + "}\n";
        return {};
    }

    /// Emits a fused Compute chain + its Store (walker execPair / 1-D
    /// runElementwiseRange contract). Per-pair block scope: chain temps
    /// have execPair lifetime and sibling pairs re-declare the same
    /// temp names — legal only in disjoint scopes (the C++ emitter's
    /// round-15 scope fix applies verbatim here).
    [[nodiscard]] Result<void> emitComputePair(const KernelNode& compute,
                                               const KernelNode& store) {
        body() += indent() + "{\n";
        ++depth;
        std::string value;
        if (compute.exprs.empty()) {
            if (multiDim) {
                // Walker execPair with an empty chain stores 0.0
                // (degenerate shape, documented).
                value = "0.0";
            } else {
                KernelExpr single;
                single.op = compute.math;
                single.a.kind = KernelOperand::Kind::ElemA;
                single.b.kind = KernelOperand::Kind::ElemB;
                MLK_TRY_VAR(rhs, apply(single, compute));
                value = rhs;
            }
        } else {
            for (std::size_t k = 0; k < compute.exprs.size(); ++k) {
                if (k >= kCudaMaxTempSlots) {
                    return err(ErrorCode::InvalidGraph,
                               "cuda emitter: compute chain exceeds the "
                               "temp-slot cap");
                }
                const KernelExpr& e = compute.exprs[k];
                const auto badTemp = [&](const KernelOperand& o) {
                    return o.kind == KernelOperand::Kind::Temp &&
                           o.index >= static_cast<int64_t>(k);
                };
                if (badTemp(e.a) || badTemp(e.b)) {
                    return err(ErrorCode::InvalidGraph,
                               "cuda emitter: temp operand is not "
                               "earlier in the chain");
                }
                MLK_TRY_VAR(rhs, apply(e, compute));
                body() += indent() + "const double mlk_t" +
                          i64(static_cast<int64_t>(k)) + " = " + rhs +
                          ";\n";
            }
            value = "mlk_t" +
                    i64(static_cast<int64_t>(compute.exprs.size()) - 1);
        }
        if (multiDim) {
            auto stored1 = emitAffineStore(store, value);
            if (!stored1.has_value()) {
                return std::unexpected<Error>(stored1.error());
            }
        } else {
            if (store.accum == AccumMode::Max) {
                return err(ErrorCode::UnsupportedCapability,
                           "cuda emitter: max-accumulate stores belong to "
                           "the multi-dim form (walker parity)");
            }
            if (varNames.empty()) {
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: store outside a loop");
            }
            auto ptr = bufRef(store.bufferOut);
            if (!ptr.has_value()) {
                return std::unexpected<Error>(ptr.error());
            }
            body() += indent() + *ptr + "[" + varNames.back() + "]" +
                      (store.accum == AccumMode::Add ? " += " : " = ") +
                      value + ";\n";
        }
        --depth;
        body() += indent() + "}\n";
        return {};
    }

    /// Sibling list with the executor's pairing discipline (multi-dim:
    /// Compute + IMMEDIATE next Store sibling; legacy: Compute + FIRST
    /// Store among the siblings).
    [[nodiscard]] Result<void> emitChildList(
        const SmallVector<uint32_t, 4>& children, const bool firstStore) {
        for (std::size_t ci = 0; ci < children.size(); ++ci) {
            const uint32_t cid = children[ci];
            if (cid >= km.nodes.size()) {
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: child id out of range");
            }
            const KernelNode& c = km.nodes[cid];
            if (c.op == KernelOp::Compute) {
                const KernelNode* store = nullptr;
                if (firstStore) {
                    for (const uint32_t sid : children) {
                        if (sid < km.nodes.size() &&
                            km.nodes[sid].op == KernelOp::Store) {
                            store = &km.nodes[sid];
                            break;
                        }
                    }
                } else if (ci + 1 < children.size() &&
                           children[ci + 1] < km.nodes.size() &&
                           km.nodes[children[ci + 1]].op ==
                               KernelOp::Store) {
                    store = &km.nodes[children[ci + 1]];
                }
                if (store != nullptr) {
                    MLK_TRYV(emitComputePair(c, *store));
                } else {
                    body() += indent() +
                              "// bare compute: no paired store (no "
                              "semantics)\n";
                }
                if (store != nullptr && !firstStore) ++ci;  // consume
                continue;
            }
            if (!multiDim) {
                body() += indent() + "// " + kernelOpName(c.op) +
                          " (no semantics in the 1-D executor)\n";
                continue;
            }
            MLK_TRYV(emitNode(cid));
        }
        return {};
    }

    /// One node of the multi-dim forest below the collapse point (walker
    /// execNode contract, serial inside the thread). No threading
    /// pragmas exist on the device path: nested parallel marks are
    /// RECORDED as serial (single-parallel-level rule — the collapse
    /// owns the grid, spec #GPU-backend), vectorHint counts are
    /// recorded (advisory; nvcc auto-vectorizes).
    [[nodiscard]] Result<void> emitNode(const uint32_t nodeId) {
        if (nodeId >= km.nodes.size()) {
            return err(ErrorCode::InvalidGraph,
                       "cuda emitter: node id out of range");
        }
        const KernelNode& n = km.nodes[nodeId];
        switch (n.op) {  // Rule 78: exhaustive
            case KernelOp::Loop: {
                if (n.step != 1) {
                    return err(ErrorCode::UnsupportedCapability,
                               "cuda emitter: non-unit loop step in a "
                               "multi-dim nest");
                }
                if (n.parallel) ++nestedParallelSerial;
                if (n.vectorHint) ++simdLoops;
                std::string beginExpr;
                if (n.beginCoeffs.empty()) {
                    beginExpr = i64(n.begin);
                } else {
                    MLK_TRY_VAR(v, affine(n.beginCoeffs, n.beginOffset));
                    beginExpr = v;
                }
                std::string endExpr;
                if (!n.endCoeffs.empty()) {
                    MLK_TRY_VAR(v, affine(n.endCoeffs, n.endOffset));
                    endExpr = v;
                } else if (n.end == constants::kKernelLoopDynamicBound &&
                           n.endBuf != constants::kInvalidId &&
                           n.endBuf < km.buffers.size() && n.endDim >= 0 &&
                           n.endDim < static_cast<int32_t>(
                                          km.buffers[n.endBuf]
                                              .dims.size())) {
                    endExpr = dimsName[n.endBuf] + "[" + i64(n.endDim) +
                              "] - 1";
                } else if (n.end == constants::kKernelLoopDynamicBound) {
                    return err(ErrorCode::UnsupportedCapability,
                               "cuda emitter: legacy dynamic bound in a "
                               "multi-dim module (bind endBuf/endDim or "
                               "constant bounds)");
                } else {
                    endExpr = i64(n.end - 1);
                }
                const std::string var = uniqueName(symbols.text(n.var));
                varNames.push_back(var);
                body() += indent() + "for (int64_t " + var + " = " +
                          beginExpr + "; " + var + " <= " + endExpr +
                          "; ++" + var + ") {\n";
                ++depth;
                MLK_TRYV(emitChildList(n.children, false));
                --depth;
                body() += indent() + "}\n";
                varNames.pop_back();
                return {};
            }
            case KernelOp::Guard: {
                if (!n.hasAffineGuard()) {
                    return err(ErrorCode::UnsupportedCapability,
                               "cuda emitter: speculative (non-affine) "
                               "guards belong to ExecutionEngine (Rule 5)");
                }
                MLK_TRY_VAR(cond, affine(n.guardCoeffs, n.guardOffset));
                body() += indent() + "if (" + cond + " == 0) {\n";
                ++depth;
                MLK_TRYV(emitChildList(n.children, false));
                --depth;
                body() += indent() + "}\n";
                return {};
            }
            case KernelOp::Compute:
                body() += indent() +
                          "// bare compute: no paired store (no semantics)\n";
                return {};
            case KernelOp::Store:
                body() += indent() + "// unconsumed store (no-op)\n";
                return {};
            case KernelOp::Load:
                body() += indent() + "// load (no-op in the artifact)\n";
                return {};
            case KernelOp::Barrier:
                body() += indent() +
                          "// barrier (no-op: single-region artifact)\n";
                return {};
            case KernelOp::Trace:
                // Rule 96: the point is RECORDED, never dropped silently.
                body() += indent() + "// [mlk:trace] runtime trace point\n";
                return {};
            case KernelOp::Call:
                return err(ErrorCode::UnsupportedCapability,
                           "cuda emitter: Call nodes must be lowered "
                           "before emission (poly.synth)",
                           121);
            case KernelOp::AllocBuffer:
            case KernelOp::CopyBuffer:
                return err(ErrorCode::UnsupportedCapability,
                           "cuda emitter: buffer management nodes are "
                           "outside the lowering/polyhedral emission "
                           "alphabet");
            case KernelOp::kCount:
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: kCount sentinel");
        }
        return err(ErrorCode::InvalidGraph, "cuda emitter: unreachable op");
    }

    /// Current emission target: the device-kernel text while walking a
    /// root, the host-wrapper text while emitting the ABI function.
    std::string* bodyTarget_{&devBody};
    [[nodiscard]] std::string& body() { return *bodyTarget_; }

    /// Interval expressions [min, max] of an affine form over the
    /// collapsed prefix vars' intervals (exact: an affine function
    /// attains its extrema at box corners; the box is the interval
    /// bound of each prefix var). Coefficients are compile-time
    /// constants, so the min/max term selection is text-time.
    [[nodiscard]] Result<std::pair<std::string, std::string>> formInterval(
        const SmallVector<int64_t, 4>& coeffs, const int64_t offset) const {
        std::string mn = i64(offset);
        std::string mx = i64(offset);
        for (std::size_t p = 0; p < coeffs.size(); ++p) {
            const int64_t c = coeffs[p];
            if (c == 0) continue;
            if (p >= prefixIntervals_.size()) {
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: collapse form references a var "
                           "outside the prefix");
            }
            const auto term = [](const int64_t k, const std::string& name) {
                return k == 1 ? name : i64(k) + "*" + name;
            };
            if (c > 0) {
                mn += " + " + term(c, prefixIntervals_[p].first);
                mx += " + " + term(c, prefixIntervals_[p].second);
            } else {
                mn += " - " + term(-c, prefixIntervals_[p].second);
                mx += " - " + term(-c, prefixIntervals_[p].first);
            }
        }
        return std::make_pair("(" + mn + ")", "(" + mx + ")");
    }

    /// The begin expression (over the prefix names) and its interval min.
    [[nodiscard]] Result<std::pair<std::string, std::string>> beginExprAndMin(
        const KernelNode& n) {
        if (n.beginCoeffs.empty()) {
            return std::make_pair(i64(n.begin), i64(n.begin));
        }
        MLK_TRY_VAR(e, affineOver(n.beginCoeffs, n.beginOffset,
                                  prefixVarNames_));
        MLK_TRY_VAR(iv, formInterval(n.beginCoeffs, n.beginOffset));
        return std::make_pair(e, iv.first);
    }

    /// The INCLUSIVE end expression (over the var stack) and its
    /// interval max (walker bound semantics: endCoeffs form >
    /// endBuf/endDim dims read > constant end-1).
    [[nodiscard]] Result<std::pair<std::string, std::string>> endExprAndMax(
        const KernelNode& n) {
        if (!n.endCoeffs.empty()) {
            MLK_TRY_VAR(e, affineOver(n.endCoeffs, n.endOffset,
                                      prefixVarNames_));
            MLK_TRY_VAR(iv, formInterval(n.endCoeffs, n.endOffset));
            return std::make_pair(e, iv.second);
        }
        if (n.end == constants::kKernelLoopDynamicBound) {
            if (n.endBuf == constants::kInvalidId ||
                n.endBuf >= km.buffers.size() || n.endDim < 0) {
                return err(ErrorCode::InvalidGraph,
                           "cuda emitter: dynamic bound without "
                           "endBuf/endDim in a collapsed level");
            }
            std::string e = dimsName[n.endBuf] + "[" + i64(n.endDim) +
                            "] - 1";
            return std::make_pair("(" + e + ")", "(" + e + ")");
        }
        return std::make_pair(i64(n.end - 1), i64(n.end - 1));
    }

    /// Collapsed-prefix var intervals (lo/hi text exprs) accumulated
    /// level by level; the interval arithmetic behind the padded grid.
    std::vector<std::pair<std::string, std::string>> prefixIntervals_{};
    /// Collapsed-prefix var NAMES (created in analysis pass 0, reused
    /// by the device emission so the two passes agree).
    SmallVector<std::string, 8> prefixVarNames_{};

    /// Emits ONE device kernel for one forest root. collapsed = the
    /// chain of parallel-marked unit-step loops peeled into the grid
    /// (possibly empty -> a one-thread serial launch, recorded).
    /// Padded grid (spec #GPU-backend: tile loops -> blocks, point
    /// loops -> threads): per-level PADDED trip counts are the exact
    /// interval max of the level's trip over the prefix box, so a flat
    /// row-major decomposition covers every valid tuple at least once;
    /// each thread then checks its var against its OWN begin/end forms
    /// and exits on padding holes. Soundness: the parallel marks prove
    /// every collapsed level's instances slab-disjoint, so any
    /// enumeration (padded or not) is deterministic; coverage is
    /// exact because the padded trip is >= every prefix tuple's real
    /// trip (interval arithmetic on affine forms) and holes return
    /// before any payload runs.
    [[nodiscard]] Result<void> emitDeviceKernel(
        const uint32_t rootId,
        const SmallVector<uint32_t, 8>& collapsed) {
        const std::string fn = "mlk_dev_" + i64(static_cast<int64_t>(
                                                  devKernelCount_++));
        devBody += "\n__global__ void " + fn + "(";
        bool first = true;
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid]) continue;
            if (!first) devBody += ",\n    ";
            const bool writable = stored[bid] ||
                                  km.buffers[bid].isOutput ||
                                  km.buffers[bid].isTemp;
            devBody += std::string(writable ? "double* " : "const double* ") +
                       ptrName[bid] + ", const int64_t* " + dimsName[bid];
            first = false;
        }
        if (!first) devBody += ",\n    ";
        devBody += "const double* mlk_scalars, int64_t mlk_n_scalars, "
                   "int* mlk_status) {\n";
        // Unused-parameter insurance: every bindable + scalars is part
        // of the uniform device signature (a cast, never a drop).
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid]) continue;
            devBody += "    (void)" + ptrName[bid] + "; (void)" +
                       dimsName[bid] + ";\n";
        }
        devBody += "    (void)mlk_scalars; (void)mlk_n_scalars;\n";

        if (!collapsed.empty()) {
            // Padded collapse — pass 0: create the prefix var names
            // once (the analysis and the device emission share them).
            prefixVarNames_.clear();
            for (const uint32_t lid : collapsed) {
                prefixVarNames_.push_back(
                    uniqueName(symbols.text(km.nodes[lid].var)));
            }
            // Pass 1: per-level padded trips + var intervals (text
            // expressions; identical on the host side because both
            // sides share the dims-arg names).
            std::vector<std::string> tmax;
            std::vector<std::string> beginE, endE;
            prefixIntervals_.clear();
            for (const uint32_t lid : collapsed) {
                const KernelNode& n = km.nodes[lid];
                MLK_TRY_VAR(bm, beginExprAndMin(n));
                MLK_TRY_VAR(em2, endExprAndMax(n));
                const std::string t = "((" + em2.second + ") - (" +
                                      bm.second + ") + 1) > 0 ? ((" +
                                      em2.second + ") - (" + bm.second +
                                      ") + 1) : 0";
                tmax.push_back(t);
                beginE.push_back(bm.first);
                endE.push_back(em2.first);
                prefixIntervals_.emplace_back(bm.second, em2.second);
            }
            // Pass 2: device text — padded trips, flat id, row-major
            // decomposition, per-thread ACTUAL range checks.
            for (std::size_t l = 0; l < tmax.size(); ++l) {
                devBody += "    const int64_t mlk_tmax" + i64(
                                           static_cast<int64_t>(l)) +
                           " = " + tmax[l] + ";\n";
            }
            std::string total = "mlk_tmax0";
            for (std::size_t l = 1; l < tmax.size(); ++l) {
                total = "(" + total + " * mlk_tmax" +
                        i64(static_cast<int64_t>(l)) + ")";
            }
            devBody += "    const int64_t mlk_total = " + total + ";\n";
            devBody += "    const int64_t mlk_flat = "
                       "(int64_t)blockIdx.x * (int64_t)blockDim.x + "
                       "(int64_t)threadIdx.x;\n";
            devBody += "    if (mlk_flat >= mlk_total) return;\n";
            devBody += "    int64_t mlk_rem = mlk_flat;\n";
            for (std::size_t l = 0; l < collapsed.size(); ++l) {
                const std::string var = prefixVarNames_[l];
                const std::string lsuf =
                    "mlk_l" + i64(static_cast<int64_t>(l));
                if (l + 1 < collapsed.size()) {
                    // Divide by the suffix product (row-major: the
                    // outermost var has the largest stride).
                    std::string suffix = "mlk_tmax" +
                                         i64(static_cast<int64_t>(l + 1));
                    for (std::size_t m = l + 2; m < collapsed.size(); ++m) {
                        suffix = "(" + suffix + " * mlk_tmax" +
                                 i64(static_cast<int64_t>(m)) + ")";
                    }
                    devBody += "    const int64_t " + lsuf +
                               " = mlk_rem / " + suffix + ";\n";
                    devBody += "    mlk_rem %= " + suffix + ";\n";
                } else {
                    devBody += "    const int64_t " + lsuf +
                               " = mlk_rem;\n";
                }
                devBody += "    int64_t " + var + " = " + beginE[l] +
                           " + " + lsuf + ";\n";
                devBody += "    if (" + var + " > " + endE[l] +
                           ") return;\n";
                varNames.push_back(var);
            }
            ++collapsedRoots;
            // The collapsed levels' parallel marks ARE the grid proof
            // (recorded as consumed); deeper marks stay serial.
            parallelMarks += collapsed.size();
            // The deepest peeled level's body executes with the
            // collapsed vars seeded on the stack.
            const KernelNode& deepest = km.nodes[collapsed.back()];
            if (deepest.step != 1) {
                return err(ErrorCode::UnsupportedCapability,
                           "cuda emitter: non-unit loop step in a "
                           "multi-dim nest");
            }
            depth = 1;
            MLK_TRYV(emitChildList(deepest.children, false));
            depth = 0;
            prefixIntervals_.clear();
            prefixVarNames_.clear();
        } else {
            // No collapse: the whole root executes in ONE thread (the
            // walker's serial semantics verbatim; recorded in the
            // header — never a silent loss of parallelism).
            ++serialRoots;
            depth = 0;
            MLK_TRYV(emitNode(rootId));
        }
        devBody += "}\n";
        varNames.clear();
        return {};
    }

    std::size_t devKernelCount_{0};

    /// Emits the host wrapper: cudaMalloc/H2D per bindable + dims,
    /// device temps from static dims (zero-initialized — the walker's
    /// temp model), per-root launches in ROOT ORDER with a
    /// synchronization between (the walker's sequential root
    /// discipline), status readback, D2H outputs, and a leak-free
    /// cleanup ladder. ABI error codes: 1 = negative store flat
    /// (device status), 2 = cuda runtime alloc/copy error, 3 = launch
    /// or sync error, 4 = no CUDA device. The driver maps every
    /// nonzero code to an error Result (the caller discards partial
    /// outputs — same observable contract as the walker's InvalidGraph).
    [[nodiscard]] Result<void> emitHostWrapper() {
        hostBody += "\nextern \"C\" int mlk_kernel(\n";
        bool first = true;
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid]) continue;
            if (!first) hostBody += ",\n";
            const bool writable = stored[bid] ||
                                  km.buffers[bid].isOutput ||
                                  km.buffers[bid].isTemp;
            hostBody += std::string("    ") +
                        (writable ? "double* " : "const double* ") +
                        ptrName[bid] + ", const int64_t* " + dimsName[bid];
            first = false;
        }
        if (!first) hostBody += ",\n";
        hostBody += "    const double* mlk_scalars, "
                    "int64_t mlk_n_scalars) {\n";
        // Cleanup ladder state: every allocation is registered; the
        // fail label frees exactly what was allocated. (Generated
        // code: the allocation count is a compile-time constant of the
        // module — bindables + their dims arrays + temps reuse the
        // bindable slots on the device + scalars + status.)
        std::size_t allocCount = 0;
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid]) continue;
            allocCount += 2;  // data + dims
        }
        allocCount += 2;  // scalars + status
        hostBody += "    void* mlk_allocs[" +
                    i64(static_cast<int64_t>(allocCount)) +
                    "] = {0};\n";
        hostBody += "    int mlk_na = 0;\n";
        hostBody += "    int mlk_code = 0;\n";
        hostBody += "    int mlk_host_status = 0;\n";
        hostBody += "    int mlk_dev_count = 0;\n";
        // ALL declarations hoisted above the first goto (C++ forbids
        // jumping over an initialized declaration in the same scope).
        hostBody += "    double* mlk_dscalars = 0;\n";
        hostBody += "    int* mlk_dstatus = 0;\n";
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid]) continue;
            hostBody += "    double* mlk_dp" + i64(bid) + " = 0;\n";
            hostBody += "    int64_t* mlk_dd" + i64(bid) + " = 0;\n";
            if (!km.buffers[bid].isTemp) {
                std::string nexpr;
                for (std::size_t d = 0; d < km.buffers[bid].dims.size();
                     ++d) {
                    const std::string term =
                        dimsName[bid] + "[" +
                        i64(static_cast<int64_t>(d)) + "]";
                    nexpr = nexpr.empty() ? term
                                          : "(" + nexpr + " * " + term +
                                                ")";
                }
                hostBody += "    int64_t mlk_n" + i64(bid) + " = (" +
                            nexpr + ") > 0 ? (" + nexpr + ") : 0;\n";
            }
        }
        hostBody += "    if (cudaGetDeviceCount(&mlk_dev_count) != "
                    "cudaSuccess || mlk_dev_count <= 0) {\n";
        hostBody += "        return 4;\n    }\n";
        // Per-bindable device allocations + H2D (the caller's dims
        // arrays are the trusted binder contract, exactly the
        // walker's). Temps: device scratch from the module's STATIC
        // dims (the ABI temp entries are accepted for uniformity and
        // ignored — documented boundary).
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid]) continue;
            const KernelBuffer& b = km.buffers[bid];
            const std::string dp = "mlk_dp" + i64(bid);
            const std::string dd = "mlk_dd" + i64(bid);
            if (b.isTemp) {
                if (b.dims.empty()) {
                    return err(ErrorCode::UnsupportedCapability,
                               "cuda emitter: temp buffer without static "
                               "dims (dynamic temp size is not "
                               "materializable in a standalone artifact)");
                }
                int64_t n = 1;
                for (const int64_t d : b.dims) {
                    if (d <= 0 ||
                        n > constants::kKernelTempElementsLimit / d) {
                        return err(ErrorCode::InvalidGraph,
                                   "cuda emitter: temp buffer element "
                                   "count out of range");
                    }
                    n *= d;
                }
                hostBody += "    if (cudaMalloc(&" + dp + ", sizeof(double) * " +
                            i64(n) + ") != cudaSuccess) { mlk_code = 2; "
                            "goto mlk_fail; }\n";
                hostBody += "    mlk_allocs[mlk_na++] = " + dp + ";\n";
                hostBody += "    if (cudaMemset(" + dp + ", 0, sizeof(double) * " +
                            i64(n) + ") != cudaSuccess) { mlk_code = 2; "
                            "goto mlk_fail; }\n";
            } else {
                if (b.dims.empty()) {
                    return err(ErrorCode::UnsupportedCapability,
                               "cuda emitter: multi-dim module binds a "
                               "buffer without static dims");
                }
                hostBody += "    if (cudaMalloc(&" + dp +
                            ", sizeof(double) * mlk_n" + i64(bid) +
                            ") != cudaSuccess) { mlk_code = 2; goto "
                            "mlk_fail; }\n";
                hostBody += "    mlk_allocs[mlk_na++] = " + dp + ";\n";
                hostBody += "    if (cudaMemcpy(" + dp + ", " + ptrName[bid] +
                            ", sizeof(double) * mlk_n" + i64(bid) +
                            ", cudaMemcpyHostToDevice) != cudaSuccess) { "
                            "mlk_code = 2; goto mlk_fail; }\n";
            }
            // Dims arrays: H2D for ALL bindables (temps too — the
            // device walk reads endBuf bounds through them uniformly).
            std::string dexpr = i64(static_cast<int64_t>(b.dims.size()));
            hostBody += "    if (cudaMalloc(&" + dd + ", sizeof(int64_t) * " +
                        dexpr + ") != cudaSuccess) { mlk_code = 2; goto "
                        "mlk_fail; }\n";
            hostBody += "    mlk_allocs[mlk_na++] = " + dd + ";\n";
            hostBody += "    if (cudaMemcpy(" + dd + ", " + dimsName[bid] +
                        ", sizeof(int64_t) * " + dexpr +
                        ", cudaMemcpyHostToDevice) != cudaSuccess) { "
                        "mlk_code = 2; goto mlk_fail; }\n";
        }
        hostBody += "    if (cudaMalloc(&mlk_dscalars, sizeof(double) * "
                    "(mlk_n_scalars > 0 ? mlk_n_scalars : 1)) != "
                    "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        hostBody += "    mlk_allocs[mlk_na++] = mlk_dscalars;\n";
        hostBody += "    if (mlk_n_scalars > 0 && cudaMemcpy(mlk_dscalars, "
                    "mlk_scalars, sizeof(double) * mlk_n_scalars, "
                    "cudaMemcpyHostToDevice) != cudaSuccess) { mlk_code = "
                    "2; goto mlk_fail; }\n";
        hostBody += "    if (cudaMalloc(&mlk_dstatus, sizeof(int)) != "
                    "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        hostBody += "    mlk_allocs[mlk_na++] = mlk_dstatus;\n";
        hostBody += "    if (cudaMemset(mlk_dstatus, 0, sizeof(int)) != "
                    "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";

        // Launches: one per root, in root order (the walker executes
        // roots sequentially in id order; same-stream launches +
        // explicit sync preserve that program order exactly).
        for (std::size_t r = 0; r < rootLaunches_.size(); ++r) {
            const RootLaunch& L = rootLaunches_[r];
            std::string args;
            bool f1 = true;
            for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
                if (!bindable[bid]) continue;
                args += (f1 ? "" : ", ") + ptrNameDev(bid) + ", " +
                        dimsNameDev(bid);
                f1 = false;
            }
            if (!f1) args += ", ";
            args += "mlk_dscalars, mlk_n_scalars, mlk_dstatus";
            hostBody += "    {\n";
            hostBody += "        const int64_t mlk_rt = " + L.totalExpr +
                        ";\n";
            hostBody += "        if (mlk_rt > 0) {\n";
            hostBody += "            const unsigned mlk_block = (unsigned)"
                        "(mlk_rt < " + i64(kCudaMaxThreadsPerBlock) +
                        " ? mlk_rt : " + i64(kCudaMaxThreadsPerBlock) +
                        ");\n";
            hostBody += "            const unsigned mlk_grid = (unsigned)"
                        "((mlk_rt + mlk_block - 1) / mlk_block);\n";
            hostBody += "            " + L.fn + "<<<mlk_grid, mlk_block>>>(" +
                        args + ");\n";
            hostBody += "            if (cudaGetLastError() != cudaSuccess)"
                        " { mlk_code = 3; goto mlk_fail; }\n";
            hostBody += "            if (cudaDeviceSynchronize() != "
                        "cudaSuccess) { mlk_code = 3; goto mlk_fail; }\n";
            hostBody += "        }\n";
            hostBody += "    }\n";
        }
        hostBody += "    if (cudaMemcpy(&mlk_host_status, mlk_dstatus, "
                    "sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) "
                    "{ mlk_code = 2; goto mlk_fail; }\n";
        hostBody += "    if (mlk_host_status != 0) {\n";
        hostBody += "        mlk_code = mlk_host_status;\n";
        hostBody += "        goto mlk_fail;\n";
        hostBody += "    }\n";
        // D2H: stored / output buffers only (the ABI's writable set).
        for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
            if (!bindable[bid] || km.buffers[bid].isTemp) continue;
            if (!(stored[bid] || km.buffers[bid].isOutput)) continue;
            hostBody += "    if (cudaMemcpy(" + ptrName[bid] +
                        ", mlk_dp" + i64(bid) + ", sizeof(double) * mlk_n" +
                        i64(bid) + ", cudaMemcpyDeviceToHost) != "
                        "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        }
        hostBody += "mlk_fail:\n";
        hostBody += "    for (int mlk_i = 0; mlk_i < mlk_na; ++mlk_i) "
                    "cudaFree(mlk_allocs[mlk_i]);\n";
        hostBody += "    return mlk_code;\n";
        hostBody += "}\n";
        return {};
    }

    /// Device-side parameter names for the launch args (the same
    /// generated pointers the wrapper allocates).
    [[nodiscard]] std::string ptrNameDev(uint32_t bid) const {
        return "mlk_dp" + i64(bid);
    }
    [[nodiscard]] std::string dimsNameDev(uint32_t bid) const {
        return "mlk_dd" + i64(bid);
    }

    struct RootLaunch {
        std::string fn{};         // __global__ function name
        std::string totalExpr{};  // host-side trip-count expression
    };
    std::vector<RootLaunch> rootLaunches_{};

    /// The collapse prefix for a root: the chain of nested parallel-
    /// marked, unit-step loops whose interior bodies are exactly one
    /// loop child. Non-rectangular levels (point loops with affine
    /// bounds over the prefix) ARE peelable — the padded grid covers
    /// them via interval-computed trip maxima (see emitDeviceKernel).
    [[nodiscard]] SmallVector<uint32_t, 8> collapseChain(
        const uint32_t rootId) const {
        SmallVector<uint32_t, 8> chain;
        uint32_t cur = rootId;
        while (cur < km.nodes.size()) {
            const KernelNode& n = km.nodes[cur];
            if (n.op != KernelOp::Loop) break;
            if (!n.parallel || n.step != 1) break;
            chain.push_back(cur);
            if (n.children.size() == 1 &&
                n.children[0] < km.nodes.size() &&
                km.nodes[n.children[0]].op == KernelOp::Loop) {
                cur = n.children[0];
                continue;
            }
            break;
        }
        return chain;
    }

    /// Host-side total trip expression for a root launch: the product
    /// of the collapsed levels' PADDED trip forms over the caller's
    /// dims arrays (1 for a serial root). The text is generated by the
    /// SAME interval machinery as the device's mlk_total, so the two
    /// sides can never diverge.
    [[nodiscard]] Result<std::string> hostTotalExpr(
        const SmallVector<uint32_t, 8>& chain) {
        if (chain.empty()) return std::string("1");
        // Rebuild the padded-trip texts (pass 0 + pass 1 of
        // emitDeviceKernel, minus the device emission). The resulting
        // text contains only constants and dims reads — never a var
        // name — so host and device totals cannot diverge.
        prefixVarNames_.clear();
        for (const uint32_t lid : chain) {
            prefixVarNames_.push_back(
                uniqueName(symbols.text(km.nodes[lid].var)));
        }
        std::vector<std::string> tmax;
        prefixIntervals_.clear();
        for (const uint32_t lid : chain) {
            const KernelNode& n = km.nodes[lid];
            MLK_TRY_VAR(bm, beginExprAndMin(n));
            MLK_TRY_VAR(em2, endExprAndMax(n));
            tmax.push_back("((" + em2.second + ") - (" + bm.second +
                           ") + 1) > 0 ? ((" + em2.second + ") - (" +
                           bm.second + ") + 1) : 0");
            prefixIntervals_.emplace_back(bm.second, em2.second);
        }
        prefixIntervals_.clear();
        prefixVarNames_.clear();
        std::string total = "(" + tmax[0] + ")";
        for (std::size_t l = 1; l < tmax.size(); ++l) {
            total = "(" + total + " * (" + tmax[l] + "))";
        }
        return total;
    }
};

/// Device "poly7" (Sin) snapshot of mlk/support/math_families.h — the
/// header stays the single source (Rule 77); the artifact inlines it to
/// remain standalone. Operation ORDER is identical to polySin so the
/// compiled artifact stays bit-exact against the walker (floor/fmod are
/// IEEE-exact operations on the device; the Horner chains are pure
/// mul/add, covered by --fmad=false).
void emitPoly7DevicePrologue(std::string& out) {
    out += "\n// Family \"poly7\" (Sin): snapshot of "
           "mlk/support/math_families.h\n";
    out += "// (Rule 77: single source; the artifact is standalone).\n";
    out += "// Verified accuracy contract: single-digit ULP vs libm on "
           "[-pi, pi] (Rule 34).\n";
    out += "__device__ static const double mlk_pio2_hi = "
           "1.5707963267948965580e+00;\n";
    out += "__device__ static const double mlk_pio2_lo = "
           "6.1232339957367660359e-17;\n";
    out += "__device__ static double mlk_poly7_sin_reduced(double x) {\n";
    out += "    const double r2 = x * x;\n";
    out += "    double p = 1.58962301576546568060e-10;\n";
    out += "    p = p * r2 + (-2.50507477628578072866e-8);\n";
    out += "    p = p * r2 + 2.75573136213857245213e-6;\n";
    out += "    p = p * r2 + (-1.98412698295895385996e-4);\n";
    out += "    p = p * r2 + 8.33333333332211858878e-3;\n";
    out += "    p = p * r2 + (-1.66666666666666307295e-1);\n";
    out += "    return x + x * r2 * p;\n";
    out += "}\n";
    out += "__device__ static double mlk_poly7_cos_reduced(double x) {\n";
    out += "    const double r2 = x * x;\n";
    out += "    double p = (-1.13585365213876817300e-11);\n";
    out += "    p = p * r2 + 2.08757530072652689750e-9;\n";
    out += "    p = p * r2 + (-2.75573142103085808917e-7);\n";
    out += "    p = p * r2 + 2.48015872890001867312e-5;\n";
    out += "    p = p * r2 + (-1.38888888888783992457e-3);\n";
    out += "    p = p * r2 + 4.16666666666666435770e-2;\n";
    out += "    return 1.0 - r2 * 0.5 + r2 * r2 * p;\n";
    out += "}\n";
    out += "__device__ static double mlk_poly7_sin(double x) {\n";
    out += "    const double nd = floor(x / 1.57079632679489661923 + "
           "0.5);\n";
    out += "    const double r = (x - nd * mlk_pio2_hi) - nd * "
           "mlk_pio2_lo;\n";
    out += "    int q = static_cast<int>(fmod(nd, 4.0));\n";
    out += "    q = q < 0 ? q + 4 : q;\n";
    out += "    switch (q) {\n";
    out += "        case 0: return mlk_poly7_sin_reduced(r);\n";
    out += "        case 1: return mlk_poly7_cos_reduced(r);\n";
    out += "        case 2: return -mlk_poly7_sin_reduced(r);\n";
    out += "        default: return -mlk_poly7_cos_reduced(r);\n";
    out += "    }\n";
    out += "}\n";
}

/// Op-set scan shared by apply()'s policy flips and the standalone
/// classification (must stay in sync — one decides emission text, the
/// other the differential gate).
[[nodiscard]] bool opIsTranscendental(MathOp op,
                                      bool poly7Sin) noexcept {
    switch (op) {  // Rule 78: exhaustive
        case MathOp::Add:
        case MathOp::Sub:
        case MathOp::Mul:
        case MathOp::Div:
        case MathOp::Neg:
        case MathOp::Sqrt:      // IEEE correctly rounded on device
        case MathOp::Rsqrt:
            return false;
        case MathOp::Sin:
            return !poly7Sin;  // the poly7 snapshot is device-exact
        case MathOp::Pow:      // non-fast form is device libm
        case MathOp::Exp:
        case MathOp::Log:
        case MathOp::Cos:
        case MathOp::Tan:
        case MathOp::Tanh:
        case MathOp::Erf:
        case MathOp::Gelu:
            return true;
        default:
            return true;  // unknown -> conservative (declared, not silent)
    }
}

}  // namespace

bool cudaArtifactBitExactPolicy(const KernelModule& kernel,
                                SymbolTable& symbols) noexcept {
    for (const KernelNode& n : kernel.nodes) {
        if (n.exprs.empty()) {
            // Legacy single-op compute (or a non-compute node: no ops).
            if (n.op != KernelOp::Compute) continue;
            const bool poly7Sin =
                n.math == MathOp::Sin && n.family != kInvalidSymbolId &&
                symbols.text(n.family) == "poly7";
            if (opIsTranscendental(n.math, poly7Sin)) {
                return false;
            }
            continue;
        }
        for (const KernelExpr& e : n.exprs) {
            // The family lives on the owning Compute node; the SAME
            // text check apply() performs decides the Sin policy.
            const bool poly7Sin =
                e.op == MathOp::Sin && n.family != kInvalidSymbolId &&
                symbols.text(n.family) == "poly7";
            if (opIsTranscendental(e.op, poly7Sin)) return false;
        }
    }
    return true;
}

Result<std::string> emitCudaSource(const KernelModule& kernel,
                                   SymbolTable& symbols) {
    if (kernel.nodes.empty()) {
        return err(ErrorCode::InvalidGraph,
                   "cannot emit empty kernel module");
    }

    CudaEmitter em{kernel, symbols};
    em.multiDim = isMultiDimModule(kernel);

    // Name tables + store-target/bindable analysis (walker binding
    // contract: input | output | temp are bindable).
    em.ptrName.resize(kernel.buffers.size());
    em.dimsName.resize(kernel.buffers.size());
    em.stored.assign(kernel.buffers.size(), false);
    em.bindable.assign(kernel.buffers.size(), false);
    // Reserved emitter names are seeded FIRST so no buffer can collide
    // with the generated scaffolding.
    for (const char* r : {"mlk_scalars", "mlk_n_scalars", "mlk_status",
                          "mlk_flat", "mlk_rem", "mlk_total", "mlk_code",
                          "mlk_na", "mlk_allocs", "mlk_fail", "mlk_rt",
                          "mlk_block", "mlk_grid", "mlk_host_status",
                          "mlk_dev_count", "mlk_dscalars", "mlk_dstatus",
                          "mlk_i"}) {
        (void)em.uniqueName(r);
    }
    for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
        const KernelBuffer& b = kernel.buffers[bid];
        em.bindable[bid] = b.isInput || b.isOutput || b.isTemp;
        if (!em.bindable[bid]) continue;
        em.ptrName[bid] = em.uniqueName(symbols.text(b.name));
        em.dimsName[bid] = em.uniqueName(em.ptrName[bid] + "_dims");
    }
    for (const KernelNode& n : kernel.nodes) {
        if (n.op == KernelOp::Store &&
            n.bufferOut < kernel.buffers.size()) {
            em.stored[n.bufferOut] = true;
        }
    }
    bool anyBindable = false;
    for (bool b : em.bindable) anyBindable = anyBindable || b;
    if (!anyBindable) {
        return err(ErrorCode::InvalidGraph,
                   "cuda emitter: module has no bindable buffers");
    }

    if (em.multiDim) {
        // Roots (unreferenced nodes) in id order — the walker's root
        // discipline. One device kernel per root, launched in order.
        std::vector<uint32_t> roots;
        for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
            bool referenced = false;
            for (const KernelNode& n : kernel.nodes) {
                for (const uint32_t c : n.children) {
                    referenced = referenced || c == i;
                }
            }
            if (!referenced) roots.push_back(i);
        }
        if (roots.size() > kCudaMaxDeviceKernels) {
            return err(ErrorCode::UnsupportedCapability,
                       "cuda emitter: module exceeds the " +
                           i64(static_cast<int64_t>(kCudaMaxDeviceKernels)) +
                           "-root device-kernel cap");
        }
        for (const uint32_t rootId : roots) {
            const SmallVector<uint32_t, 8> chain = em.collapseChain(rootId);
            const std::size_t fnIdx = em.devKernelCount_;
            auto emitted = em.emitDeviceKernel(rootId, chain);
            if (!emitted.has_value()) {
                return std::unexpected<Error>(emitted.error());
            }
            MLK_TRY_VAR(total, em.hostTotalExpr(chain));
            em.rootLaunches_.push_back(CudaEmitter::RootLaunch{
                "mlk_dev_" + i64(static_cast<int64_t>(fnIdx)), total});
        }
        auto wrapper = em.emitHostWrapper();
        if (!wrapper.has_value()) {
            return std::unexpected<Error>(wrapper.error());
        }
    } else {
        // Legacy 1-D form: exactly ONE root loop (the 1-D executor's
        // single-range contract), one thread per element.
        uint32_t rootId = constants::kInvalidId;
        std::size_t rootCount = 0;
        for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
            bool referenced = false;
            for (const KernelNode& n : kernel.nodes) {
                for (const uint32_t c : n.children) {
                    referenced = referenced || c == i;
                }
            }
            if (!referenced) {
                ++rootCount;
                rootId = i;
            }
        }
        if (rootCount != 1 || rootId >= kernel.nodes.size() ||
            kernel.nodes[rootId].op != KernelOp::Loop) {
            return err(ErrorCode::UnsupportedCapability,
                       "cuda emitter: the legacy 1-D form requires "
                       "exactly one root loop (the 1-D executor's "
                       "single-range contract)");
        }
        const KernelNode& root = kernel.nodes[rootId];
        const std::string vi = em.uniqueName(symbols.text(root.var));
        em.devBody += "\n__global__ void mlk_dev_0(";
        bool first = true;
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (!em.bindable[bid]) continue;
            if (!first) em.devBody += ", ";
            em.devBody += std::string(
                              em.stored[bid] || kernel.buffers[bid].isOutput
                                  ? "double* "
                                  : "const double* ") +
                          em.ptrName[bid];
            first = false;
        }
        em.devBody += ", int64_t n, const double* mlk_scalars, "
                      "int64_t mlk_n_scalars, int* mlk_status) {\n";
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (!em.bindable[bid]) continue;
            em.devBody += "    (void)" + em.ptrName[bid] + ";\n";
        }
        em.devBody += "    (void)mlk_scalars; (void)mlk_n_scalars;\n";
        // One thread per element over the caller's element count; the
        // static range guard mirrors the 1-D executor's loop bounds.
        std::string endBound = "n";
        if (root.end != constants::kKernelLoopDynamicBound) {
            endBound = i64(root.end);
        }
        em.devBody += "    const int64_t " + vi +
                      " = (int64_t)blockIdx.x * (int64_t)blockDim.x + "
                      "(int64_t)threadIdx.x;\n";
        em.devBody += "    if (" + vi + " < " + i64(root.begin) + " || " +
                      vi + " >= " + endBound + ") return;\n";
        em.varNames.push_back(vi);
        em.depth = 1;
        auto walk = em.emitChildList(root.children, true);
        if (!walk.has_value()) {
            return std::unexpected<Error>(walk.error());
        }
        em.depth = 0;
        em.varNames.pop_back();
        em.devBody += "}\n";
        em.collapsedRoots = 1;  // one thread per element (grid-mapped)
        // Host wrapper (legacy ABI: ptrs + n + scalars pair).
        em.hostBody += "\nextern \"C\" void mlk_kernel(\n";
        first = true;
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (!em.bindable[bid]) continue;
            if (!first) em.hostBody += ",\n";
            em.hostBody += std::string("    ") +
                           (em.stored[bid] ||
                                    kernel.buffers[bid].isOutput
                                ? "double* "
                                : "const double* ") +
                           em.ptrName[bid];
            first = false;
        }
        em.hostBody += ",\n    int64_t n";
        em.hostBody += ",\n    const double* mlk_scalars, "
                       "int64_t mlk_n_scalars) {\n";
        em.hostBody += "    void* mlk_allocs[" +
                       i64(static_cast<int64_t>(
                           std::count(em.bindable.begin(),
                                      em.bindable.end(), true) +
                           2)) +
                       "] = {0};\n";
        em.hostBody += "    int mlk_na = 0;\n";
        em.hostBody += "    int mlk_code = 0;\n";
        em.hostBody += "    int mlk_host_status = 0;\n";
        em.hostBody += "    double* mlk_dscalars = 0;\n";
        em.hostBody += "    int* mlk_dstatus = 0;\n";
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (!em.bindable[bid]) continue;
            em.hostBody += "    double* mlk_dp" + i64(bid) + " = 0;\n";
        }
        em.hostBody += "    int mlk_dev_count = 0;\n";
        em.hostBody += "    if (cudaGetDeviceCount(&mlk_dev_count) != "
                       "cudaSuccess || mlk_dev_count <= 0) {\n";
        em.hostBody += "        mlk_code = 4; goto mlk_fail;\n    }\n";
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (!em.bindable[bid]) continue;
            em.hostBody += "    if (cudaMalloc(&mlk_dp" + i64(bid) +
                           ", sizeof(double) * (n > 0 ? n : 1)) != "
                           "cudaSuccess) { mlk_code = 2; goto mlk_fail; "
                           "}\n";
            em.hostBody += "    mlk_allocs[mlk_na++] = mlk_dp" + i64(bid) +
                           ";\n";
            em.hostBody += "    if (cudaMemcpy(mlk_dp" + i64(bid) + ", " +
                           em.ptrName[bid] + ", sizeof(double) * n, "
                           "cudaMemcpyHostToDevice) != cudaSuccess) { "
                           "mlk_code = 2; goto mlk_fail; }\n";
        }
        em.hostBody += "    if (cudaMalloc(&mlk_dscalars, sizeof(double) * "
                       "(mlk_n_scalars > 0 ? mlk_n_scalars : 1)) != "
                       "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        em.hostBody += "    mlk_allocs[mlk_na++] = mlk_dscalars;\n";
        em.hostBody += "    if (mlk_n_scalars > 0 && cudaMemcpy(mlk_dscalars, "
                       "mlk_scalars, sizeof(double) * mlk_n_scalars, "
                       "cudaMemcpyHostToDevice) != cudaSuccess) { mlk_code "
                       "= 2; goto mlk_fail; }\n";
        em.hostBody += "    if (cudaMalloc(&mlk_dstatus, sizeof(int)) != "
                       "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        em.hostBody += "    mlk_allocs[mlk_na++] = mlk_dstatus;\n";
        em.hostBody += "    if (cudaMemset(mlk_dstatus, 0, sizeof(int)) != "
                       "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        {
            std::string args;
            first = true;
            for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
                if (!em.bindable[bid]) continue;
                args += (first ? "" : ", ") + em.ptrNameDev(bid);
                first = false;
            }
            args += ", n, mlk_dscalars, mlk_n_scalars, mlk_dstatus";
            em.hostBody += "    if (n > 0) {\n";
            em.hostBody += "        const unsigned mlk_block = (unsigned)"
                           "(n < " + i64(kCudaMaxThreadsPerBlock) +
                           " ? n : " + i64(kCudaMaxThreadsPerBlock) +
                           ");\n";
            em.hostBody += "        const unsigned mlk_grid = (unsigned)"
                           "((n + mlk_block - 1) / mlk_block);\n";
            em.hostBody += "        mlk_dev_0<<<mlk_grid, mlk_block>>>(" +
                           args + ");\n";
            em.hostBody += "        if (cudaGetLastError() != cudaSuccess)"
                           " { mlk_code = 3; goto mlk_fail; }\n";
            em.hostBody += "        if (cudaDeviceSynchronize() != "
                           "cudaSuccess) { mlk_code = 3; goto mlk_fail; "
                           "}\n";
            em.hostBody += "    }\n";
        }
        em.hostBody += "    if (cudaMemcpy(&mlk_host_status, mlk_dstatus, "
                       "sizeof(int), cudaMemcpyDeviceToHost) != "
                       "cudaSuccess) { mlk_code = 2; goto mlk_fail; }\n";
        em.hostBody += "    if (mlk_host_status != 0) { mlk_code = "
                       "mlk_host_status; goto mlk_fail; }\n";
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (!em.bindable[bid]) continue;
            if (!(em.stored[bid] || kernel.buffers[bid].isOutput)) {
                continue;
            }
            em.hostBody += "    if (cudaMemcpy(" + em.ptrName[bid] +
                           ", mlk_dp" + i64(bid) + ", sizeof(double) * n, "
                           "cudaMemcpyDeviceToHost) != cudaSuccess) { "
                           "mlk_code = 2; goto mlk_fail; }\n";
        }
        em.hostBody += "mlk_fail:\n";
        em.hostBody += "    for (int mlk_i = 0; mlk_i < mlk_na; ++mlk_i) "
                       "cudaFree(mlk_allocs[mlk_i]);\n";
        em.hostBody += "    (void)mlk_code;\n";
        em.hostBody += "}\n";
    }

    // Prologue assembly: header comments (Rule 148 recorded marks +
    // the declared exactness policy + the ABI error codes), includes,
    // optional poly7 device snapshot, device kernels, host wrapper.
    std::string out;
    const std::string kernelName =
        kernel.name != kInvalidSymbolId
            ? symbols.text(kernel.name)
            : std::string("<anonymous>");
    out += "// Generated by MLK+ (mlk_backend_cpu::emitCudaSource).\n";
    out += "// Kernel: " + kernelName +
           "  hash: " + std::to_string(kernel.hash()) + "\n";
    out += "// Semantics mirror the buffer executor "
           "(runtime/src/execution/kernel_buffers.cpp); the grid comes "
           "from the scheduler's PROVEN parallel marks "
           "(docs/polyhedral_spec.md #GPU-backend).\n";
    out += "// Rule 148 recorded: device kernels=" +
           i64(static_cast<int64_t>(em.collapsedRoots +
                                    em.serialRoots)) +
           " (grid-collapsed=" +
           i64(static_cast<int64_t>(em.collapsedRoots)) +
           ", single-thread=" +
           i64(static_cast<int64_t>(em.serialRoots)) +
           "), parallel marks=" +
           i64(static_cast<int64_t>(em.parallelMarks +
                                    em.nestedParallelSerial)) +
           " (below collapse, serial=" +
           i64(static_cast<int64_t>(em.nestedParallelSerial)) +
           "), simd hints=" + i64(static_cast<int64_t>(em.simdLoops)) +
           ".\n";
    out += "// Exactness policy: " +
           std::string(em.transcendental
                           ? "ULP-BOUNDED (device-libm transcendental "
                             "present; the differential vs the walker is "
                             "measured and reported, never asserted, "
                             "never silently relaxed)"
                           : "bit-exact (IEEE mul/add/div/sqrt with "
                             "--fmad=false; the poly7 snapshot is pure "
                             "mul/add)") +
           ".\n";
    out += "// Build contract: nvcc --fmad=false (separate mul/add — no "
           "contraction, Rules 33/90; -prec-div=true default kept).\n";
    out += "// ABI error codes: 0 ok; 1 negative store flat; 2 cuda "
           "runtime alloc/copy; 3 launch/sync; 4 no CUDA device.\n";
    out += "// Temp policy: device scratch materialized from the "
           "module's static temp dims, zero-initialized (the walker's "
           "temp model); ABI temp table entries are accepted for "
           "uniformity and ignored.\n";
    out += "#include <cstdint>\n";
    if (em.needsPoly7) emitPoly7DevicePrologue(out);
    out += "\n";
    out += em.devBody;
    out += em.hostBody;
    return out;
}

}  // namespace mlk
