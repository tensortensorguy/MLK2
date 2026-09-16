// C++ source emitter implementation (see header for the ABI contract).
//
// Semantics mirror runtime/src/execution/kernel_buffers.cpp EXACTLY:
// the emitted artifact is the compiled form of the same executor
// contract (docs/kernel_abi.md), so a differential test between the
// buffer walker and the compiled artifact is meaningful — and is
// exercised in tests/unit/unit_poly.cpp (cpp_backend_* tests).
#include "mlk/backend/cpp_emitter.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mlk/core/constants.h"

namespace mlk {

namespace {

/// Same temp-slot cap as the executor (Rule 27: named, documented).
inline constexpr std::size_t kEmitterMaxTempSlots = 64;

[[nodiscard]] bool isUnaryOp(MathOp op) noexcept {
    return op == MathOp::Neg || op == MathOp::Exp || op == MathOp::Log ||
           op == MathOp::Sin || op == MathOp::Cos || op == MathOp::Tan ||
           op == MathOp::Tanh || op == MathOp::Sqrt ||
           op == MathOp::Rsqrt || op == MathOp::Erf || op == MathOp::Gelu;
}

[[nodiscard]] std::string i64(const int64_t v) { return std::to_string(v); }

/// Exact double literal (round-trippable; guarantees a float literal
/// form so integer-looking values stay doubles in the artifact).
[[nodiscard]] std::string f64(const double d) {
    if (std::isnan(d)) return "std::nan(\"\")";
    if (std::isinf(d)) return d > 0.0 ? "INFINITY" : "-INFINITY";
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.17g", d);
    std::string s = buf;
    if (s.find_first_of(".eEnN") == std::string::npos) s += ".0";
    return s;
}

/// C identifier sanitizer: keeps [A-Za-z0-9_], prefixes a leading digit,
/// guards a small reserved set. De-collision runs through the emitter's
/// used-name pool.
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

struct CppEmitter {
    const KernelModule& km;
    SymbolTable& symbols;
    bool multiDim{false};
    bool needsScalars{false};
    bool needsPoly7{false};
    bool legacyUsesDynamicN{false};
    std::size_t parallelLoops{0};
    std::size_t simdLoops{0};
    std::string body{};  // signature + everything between the braces
    int depth{0};
    SmallVector<std::string, 8> varNames{};  // per-depth emitted names
    std::vector<std::string> used{};         // uniquification pool
    std::vector<std::string> ptrName{};      // per buffer id
    std::vector<std::string> dimsName{};     // per buffer id
    std::vector<bool> stored{};              // per buffer id: store target

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

    /// Affine form over the enclosing var stack (walker flatIndex
    /// contract: sum coeffs[p]*var[p] + offset). Terms keep explicit
    /// parentheses; the whole form is parenthesized.
    [[nodiscard]] Result<std::string> affine(
        const SmallVector<int64_t, 4>& coeffs, const int64_t offset) const {
        std::string s;
        bool any = false;
        for (std::size_t p = 0; p < coeffs.size(); ++p) {
            const int64_t c = coeffs[p];
            if (c == 0) continue;  // zero coefficients never bind a var
            if (p >= varNames.size()) {
                return err(ErrorCode::InvalidGraph,
                           "emitter: affine form references an unbound "
                           "loop var");
            }
            std::string term;
            if (c == 1) {
                term = varNames[p];
            } else if (c == -1) {
                term = "-" + varNames[p];
            } else {
                term = i64(c) + "*" + varNames[p];
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

    /// Validated buffer pointer name for an operand/store id.
    [[nodiscard]] Result<std::string> bufRef(const uint32_t bid) const {
        if (bid >= km.buffers.size()) {
            return err(ErrorCode::InvalidGraph,
                       "emitter: buffer id out of range");
        }
        const KernelBuffer& b = km.buffers[bid];
        if (b.isTemp) {
            return err(ErrorCode::UnsupportedCapability,
                       "emitter: temp buffers are not supported yet in "
                       "native artifacts (roadmap; round-17)");
        }
        if (!b.isInput && !b.isOutput) {
            return err(ErrorCode::InvalidGraph,
                       "emitter: reference to an unbindable buffer "
                       "(neither input nor output)");
        }
        return ptrName[bid];
    }

    /// One scalar operand expression (walker evalOne contract). The
    /// owning Compute node provides the legacy ElemA/ElemB buffer ids.
    [[nodiscard]] Result<std::string> operand(const KernelOperand& o,
                                              const KernelNode& compute) {
        switch (o.kind) {  // Rule 78: exhaustive
            case KernelOperand::Kind::Const:
                return f64(o.constValue);
            case KernelOperand::Kind::ElemA:
            case KernelOperand::Kind::ElemB: {
                // Multi-dim computes reject legacy operands exactly like
                // the walker's execPair; in the legacy 1-D form they are
                // buffer[i] with i the single enclosing loop var.
                if (multiDim) {
                    return err(ErrorCode::InvalidGraph,
                               "emitter: legacy element operand in a "
                               "multi-dim compute");
                }
                if (varNames.empty()) {
                    return err(ErrorCode::InvalidGraph,
                               "emitter: legacy operand outside a loop");
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
                needsScalars = true;
                return "((" + i64(o.index) + " < mlk_n_scalars) ? "
                            "mlk_scalars[" + i64(o.index) + "] : 0.0)";
            }
            case KernelOperand::Kind::Temp: {
                if (o.index < 0 ||
                    static_cast<std::size_t>(o.index) >=
                        kEmitterMaxTempSlots) {
                    return err(ErrorCode::InvalidGraph,
                               "emitter: temp index out of range");
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
                   "emitter: unknown operand kind");
    }

    /// One scalar op application (walker apply() contract verbatim —
    /// including the singularity-as-value special cases and the
    /// Pow(2,2) -> a*a fast form; Rule 93: mathematical exceptions are
    /// VALUES, not errors).
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
        // Family dispatch: the walker applies the "poly7" family to Sin
        // only (resolveFamily + apply); mirror exactly.
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
                return "((" + b + ") != 0.0 ? (" + *a + ") / (" + b +
                       ") : 0.0)";
            case MathOp::Neg: return "-(" + *a + ")";
            case MathOp::Pow:
                return "(((" + *a + ") == 2.0 && (" + b + ") == 2.0) ? (" +
                       *a + ") * (" + *a + ") : std::pow((" + *a + "), (" +
                       b + ")))";
            case MathOp::Exp: return "std::exp(" + *a + ")";
            case MathOp::Log:
                return "((" + *a + ") > 0.0 ? std::log(" + *a + ") : 0.0)";
            case MathOp::Sin:
                return poly7Sin ? "mlk_poly7_sin(" + *a + ")"
                                : "std::sin(" + *a + ")";
            case MathOp::Cos: return "std::cos(" + *a + ")";
            case MathOp::Tan: return "std::tan(" + *a + ")";
            case MathOp::Tanh: return "std::tanh(" + *a + ")";
            case MathOp::Sqrt:
                return "((" + *a + ") >= 0.0 ? std::sqrt(" + *a +
                       ") : 0.0)";
            case MathOp::Rsqrt:
                return "((" + *a + ") > 0.0 ? 1.0 / std::sqrt(" + *a +
                       ") : 0.0)";
            case MathOp::Erf: return "std::erf(" + *a + ")";
            case MathOp::Gelu:
                return "0.5 * (" + *a +
                       ") * (1.0 + std::tanh(0.7978845608028654 * ((" + *a +
                       ") + 0.044715 * (" + *a + ") * (" + *a + ") * (" +
                       *a + "))))";
            default:
                // Non-realizable ops are rejected at lowering time
                // (isScalarRealizable); reaching here is a contract bug
                // (the walker degrades to 0.0; the emitter refuses).
                return err(ErrorCode::InvalidGraph,
                           "emitter: non-realizable math op");
        }
    }

    /// Emits a fused Compute chain + its Store (walker execPair / 1-D
    /// runElementwiseRange contract).
    [[nodiscard]] Result<void> emitComputePair(const KernelNode& compute,
                                               const KernelNode& store) {
        // Per-pair block scope: chain temps have execPair lifetime
        // (the executor's temp slots are per pair), and sibling pairs
        // at the same loop level re-declare the same temp names —
        // legal only in disjoint scopes. The guarded re-entry form
        // accidentally provided this scope through the guard's if
        // block; the piecewise-split form emits sibling pairs
        // unguarded, so the scope is explicit now. (Error returns
        // leave depth bumped: a failed emission discards the text.)
        body += indent() + "{\n";
        ++depth;
        std::string value;
        if (compute.exprs.empty()) {
            if (multiDim) {
                // Walker execPair with an empty chain reads a
                // out-of-range temp slot and stores 0.0 — mirrored
                // (degenerate shape, documented).
                value = "0.0";
            } else {
                // Legacy single-op compute: one op over ElemA (+ElemB
                // for binary ops) — the 1-D fast path.
                KernelExpr single;
                single.op = compute.math;
                single.a.kind = KernelOperand::Kind::ElemA;
                single.b.kind = KernelOperand::Kind::ElemB;
                MLK_TRY_VAR(rhs, apply(single, compute));
                value = rhs;
            }
        } else {
            for (std::size_t k = 0; k < compute.exprs.size(); ++k) {
                if (k >= kEmitterMaxTempSlots) {
                    return err(ErrorCode::InvalidGraph,
                               "emitter: compute chain exceeds the "
                               "temp-slot cap");
                }
                const KernelExpr& e = compute.exprs[k];
                // SSA over temps: a Temp operand references earlier
                // slots (kernel_ir.h contract; the executor reads the
                // same slots — emission enforces the contract).
                const auto badTemp = [&](const KernelOperand& o) {
                    return o.kind == KernelOperand::Kind::Temp &&
                           o.index >= static_cast<int64_t>(k);
                };
                if (badTemp(e.a) || badTemp(e.b)) {
                    return err(ErrorCode::InvalidGraph,
                               "emitter: temp operand is not earlier in "
                               "the chain");
                }
                MLK_TRY_VAR(rhs, apply(e, compute));
                body += indent() + "const double mlk_t" +
                        i64(static_cast<int64_t>(k)) + " = " + rhs + ";\n";
            }
            value = "mlk_t" +
                    i64(static_cast<int64_t>(compute.exprs.size()) - 1);
        }
        auto ptr = bufRef(store.bufferOut);
        if (!ptr.has_value()) {
            return std::unexpected<Error>(ptr.error());
        }
        if (store.accum == AccumMode::Max) {
            // Overwriting here would be silently WRONG (the Max store is
            // a read-modify-write reduction); reject until the native
            // artifact story for temps/Max lands (round-17 roadmap).
            return err(ErrorCode::UnsupportedCapability,
                       "emitter: max-accumulate stores are not supported "
                       "yet (roadmap; round-17)");
        }
        std::string target;
        if (multiDim) {
            MLK_TRY_VAR(flat,
                        affine(store.outIndexCoeffs, store.outIndexOffset));
            target = *ptr + "[" + flat + "]";
        } else {
            if (varNames.empty()) {
                return err(ErrorCode::InvalidGraph,
                           "emitter: store outside a loop");
            }
            target = *ptr + "[" + varNames.back() + "]";
        }
        body += indent() + target +
                (store.accum == AccumMode::Add ? " += " : " = ") +
                value + ";\n";
        --depth;
        body += indent() + "}\n";
        return {};
    }

    /// Sibling list with the executor's pairing discipline: multi-dim
    /// pairs a Compute with the IMMEDIATE next sibling when it is a
    /// Store (walker execChildList); the legacy 1-D form pairs every
    /// Compute with the FIRST Store among the siblings (1-D executor)
    /// and ignores every other child op.
    [[nodiscard]] Result<void> emitChildList(
        const SmallVector<uint32_t, 4>& children, const bool firstStore) {
        for (std::size_t ci = 0; ci < children.size(); ++ci) {
            const uint32_t cid = children[ci];
            if (cid >= km.nodes.size()) {
                return err(ErrorCode::InvalidGraph,
                           "emitter: child id out of range");
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
                    // Walker: a bare Compute has no semantics at this
                    // point (needs its paired Store).
                    body += indent() +
                            "// bare compute: no paired store (no "
                            "semantics)\n";
                }
                if (store != nullptr && !firstStore) ++ci;  // consume
                continue;
            }
            if (!multiDim) {
                // The 1-D executor inspects loop children for Compute
                // only; everything else carries no semantics there.
                body += indent() + "// " + kernelOpName(c.op) +
                        " (no semantics in the 1-D executor)\n";
                continue;
            }
            MLK_TRYV(emitNode(cid));
        }
        return {};
    }

    /// One node of the multi-dim forest (walker execNode contract).
    [[nodiscard]] Result<void> emitNode(const uint32_t nodeId) {
        if (nodeId >= km.nodes.size()) {
            return err(ErrorCode::InvalidGraph,
                       "emitter: node id out of range");
        }
        const KernelNode& n = km.nodes[nodeId];
        switch (n.op) {  // Rule 78: exhaustive
            case KernelOp::Loop: {
                if (n.step != 1) {
                    return err(ErrorCode::UnsupportedCapability,
                               "emitter: non-unit loop step in a "
                               "multi-dim nest");
                }
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
                    // The walker degrades to io.elements here; a
                    // standalone artifact has no binder, so this module
                    // shape must be rejected (documented boundary).
                    return err(ErrorCode::UnsupportedCapability,
                               "emitter: legacy dynamic bound in a "
                               "multi-dim module (bind endBuf/endDim or "
                               "constant bounds)");
                } else {
                    endExpr = i64(n.end - 1);
                }
                if (n.parallel || n.vectorHint) {
                    body += indent() + "#ifdef _OPENMP\n";
                    if (n.parallel && n.vectorHint) {
                        body += indent() +
                                "#pragma omp parallel for simd "
                                "schedule(static)\n";
                        ++parallelLoops;
                        ++simdLoops;
                    } else if (n.parallel) {
                        body += indent() +
                                "#pragma omp parallel for "
                                "schedule(static)\n";
                        ++parallelLoops;
                    } else {
                        body += indent() + "#pragma omp simd\n";
                        ++simdLoops;
                    }
                    body += indent() + "#endif\n";
                }
                const std::string var = uniqueName(symbols.text(n.var));
                varNames.push_back(var);
                body += indent() + "for (int64_t " + var + " = " +
                        beginExpr + "; " + var + " <= " + endExpr + "; ++" +
                        var + ") {\n";
                ++depth;
                MLK_TRYV(emitChildList(n.children, false));
                --depth;
                body += indent() + "}\n";
                varNames.pop_back();
                return {};
            }
            case KernelOp::Guard: {
                if (!n.hasAffineGuard()) {
                    return err(ErrorCode::UnsupportedCapability,
                               "emitter: speculative (non-affine) guards "
                               "belong to ExecutionEngine (Rule 5)");
                }
                MLK_TRY_VAR(cond, affine(n.guardCoeffs, n.guardOffset));
                body += indent() + "if (" + cond + " == 0) {\n";
                ++depth;
                MLK_TRYV(emitChildList(n.children, false));
                --depth;
                body += indent() + "}\n";
                return {};
            }
            case KernelOp::Compute:
                // Paired by emitChildList; a bare compute reaching here
                // has no paired store (walker: no semantics).
                body += indent() +
                        "// bare compute: no paired store (no semantics)\n";
                return {};
            case KernelOp::Store:
                // Unconsumed store (walker: no-op).
                body += indent() + "// unconsumed store (no-op)\n";
                return {};
            case KernelOp::Load:
                body += indent() + "// load (no-op in the artifact)\n";
                return {};
            case KernelOp::Barrier:
                body += indent() +
                        "// barrier (no-op: single-region artifact)\n";
                return {};
            case KernelOp::Trace:
                // Rule 96: tracing must remain correct — the runtime
                // hook has no artifact counterpart, so the point is
                // RECORDED, never dropped silently.
                body += indent() + "// [mlk:trace] runtime trace point\n";
                return {};
            case KernelOp::Call:
                return err(ErrorCode::UnsupportedCapability,
                           "emitter: Call nodes must be lowered before "
                           "emission (poly.synth)",
                           121);
            case KernelOp::AllocBuffer:
            case KernelOp::CopyBuffer:
                return err(ErrorCode::UnsupportedCapability,
                           "emitter: buffer management nodes are outside "
                           "the lowering/polyhedral emission alphabet");
            case KernelOp::kCount:
                return err(ErrorCode::InvalidGraph,
                           "emitter: kCount sentinel");
        }
        return err(ErrorCode::InvalidGraph, "emitter: unreachable op");
    }
};

void emitPoly7Prologue(std::string& out) {
    // Family "poly7" (Sin): a snapshot of mlk/support/math_families.h —
    // the header stays the single source (Rule 77); the artifact inlines
    // it to remain standalone. The operation ORDER below is identical to
    // polySin so the compiled artifact is bit-exact against the walker.
    out += "\n// Family \"poly7\" (Sin): snapshot of "
           "mlk/support/math_families.h\n";
    out += "// (Rule 77: single source; the artifact is standalone).\n";
    out += "// Verified accuracy contract: single-digit ULP vs libm on "
           "[-pi, pi] (Rule 34).\n";
    out += "static const double mlk_pio2_hi = "
           "1.5707963267948965580e+00;\n";
    out += "static const double mlk_pio2_lo = "
           "6.1232339957367660359e-17;\n";
    out += "static double mlk_poly7_sin_reduced(double x) {\n";
    out += "    const double r2 = x * x;\n";
    out += "    double p = 1.58962301576546568060e-10;\n";
    out += "    p = p * r2 + (-2.50507477628578072866e-8);\n";
    out += "    p = p * r2 + 2.75573136213857245213e-6;\n";
    out += "    p = p * r2 + (-1.98412698295895385996e-4);\n";
    out += "    p = p * r2 + 8.33333333332211858878e-3;\n";
    out += "    p = p * r2 + (-1.66666666666666307295e-1);\n";
    out += "    return x + x * r2 * p;\n";
    out += "}\n";
    out += "static double mlk_poly7_cos_reduced(double x) {\n";
    out += "    const double r2 = x * x;\n";
    out += "    double p = (-1.13585365213876817300e-11);\n";
    out += "    p = p * r2 + 2.08757530072652689750e-9;\n";
    out += "    p = p * r2 + (-2.75573142103085808917e-7);\n";
    out += "    p = p * r2 + 2.48015872890001867312e-5;\n";
    out += "    p = p * r2 + (-1.38888888888783992457e-3);\n";
    out += "    p = p * r2 + 4.16666666666666435770e-2;\n";
    out += "    return 1.0 - r2 * 0.5 + r2 * r2 * p;\n";
    out += "}\n";
    out += "static double mlk_poly7_sin(double x) {\n";
    out += "    const double nd = std::floor(x / 1.57079632679489661923 + "
           "0.5);\n";
    out += "    const double r = (x - nd * mlk_pio2_hi) - nd * "
           "mlk_pio2_lo;\n";
    out += "    int q = static_cast<int>(std::fmod(nd, 4.0));\n";
    out += "    q = q < 0 ? q + 4 : q;\n";
    out += "    switch (q) {\n";
    out += "        case 0: return mlk_poly7_sin_reduced(r);\n";
    out += "        case 1: return mlk_poly7_cos_reduced(r);\n";
    out += "        case 2: return -mlk_poly7_sin_reduced(r);\n";
    out += "        default: return -mlk_poly7_cos_reduced(r);\n";
    out += "    }\n";
    out += "}\n";
}

}  // namespace

bool isMultiDimModule(const KernelModule& kernel) noexcept {
    for (const KernelNode& n : kernel.nodes) {
        if (n.accum != AccumMode::None || !n.beginCoeffs.empty() ||
            !n.endCoeffs.empty() ||
            (n.end == constants::kKernelLoopDynamicBound &&
             n.endBuf != constants::kInvalidId)) {
            return true;
        }
        if (n.hasAffineStore() ||
            (n.op == KernelOp::Guard && n.hasAffineGuard())) {
            return true;
        }
        for (const KernelExpr& e : n.exprs) {
            if (e.a.kind == KernelOperand::Kind::ElemIdx ||
                e.b.kind == KernelOperand::Kind::ElemIdx) {
                return true;
            }
        }
    }
    return false;
}

Result<std::string> emitCppSource(const KernelModule& kernel,
                                  SymbolTable& symbols) {
    if (kernel.nodes.empty()) {
        return err(ErrorCode::InvalidGraph,
                   "cannot emit empty kernel module");
    }

    CppEmitter em{kernel, symbols};
    em.multiDim = isMultiDimModule(kernel);

    // Buffer name table + store-target analysis.
    em.ptrName.resize(kernel.buffers.size());
    em.dimsName.resize(kernel.buffers.size());
    em.stored.assign(kernel.buffers.size(), false);
    for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
        em.ptrName[bid] =
            em.uniqueName(symbols.text(kernel.buffers[bid].name));
        em.dimsName[bid] = em.uniqueName(em.ptrName[bid] + "_dims");
    }
    for (const KernelNode& n : kernel.nodes) {
        if (n.op == KernelOp::Store &&
            n.bufferOut < kernel.buffers.size()) {
            em.stored[n.bufferOut] = true;
        }
    }

    // Signature.
    em.body += "\nextern \"C\" ";
    if (em.multiDim) {
        em.body += "int mlk_kernel(\n";
        bool first = true;
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            const KernelBuffer& b = kernel.buffers[bid];
            if (!b.isInput && !b.isOutput) continue;
            if (!first) em.body += ",\n";
            const bool writable = em.stored[bid] || b.isOutput;
            em.body += std::string("    ") +
                       (writable ? "double* " : "const double* ") +
                       em.ptrName[bid] + ", const int64_t* " +
                       em.dimsName[bid];
            first = false;
        }
        if (first) {
            return err(ErrorCode::InvalidGraph,
                       "emitter: module has no bindable buffers");
        }
        // Fixed ABI: the scalars pair is ALWAYS last (see header) — an
        // empty table degrades reads to 0.0 exactly like the walker.
        em.body += ",\n    const double* mlk_scalars, "
                   "int64_t mlk_n_scalars";
        em.body += ") {\n";
    } else {
        em.body += "void mlk_kernel(\n";
        bool first = true;
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            const KernelBuffer& b = kernel.buffers[bid];
            if (!b.isInput && !b.isOutput) continue;
            if (!first) em.body += ",\n";
            em.body += std::string("    ") +
                       (b.isOutput || em.stored[bid] ? "double* "
                                                     : "const double* ") +
                       em.ptrName[bid];
            first = false;
        }
        if (first) {
            return err(ErrorCode::InvalidGraph,
                       "emitter: module has no bindable buffers");
        }
        em.body += ",\n    int64_t n";
        // Fixed ABI: the scalars pair is ALWAYS last (see header).
        em.body += ",\n    const double* mlk_scalars, "
                   "int64_t mlk_n_scalars";
        em.body += ") {\n";
    }

    // Unused-parameter insurance: dims arrays and scalar slots are part
    // of the fixed ABI even when a particular module never reads them
    // (consumer toolchains may run -Wunused-parameter -Werror). The
    // legacy 1-D signature carries no dims arrays — only the multi-dim
    // form needs the (void) silencers for them.
    if (em.multiDim) {
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            if (kernel.buffers[bid].isInput ||
                kernel.buffers[bid].isOutput) {
                em.body += "    (void)" + em.dimsName[bid] + ";\n";
            }
        }
    }
    em.body += "    (void)mlk_scalars; (void)mlk_n_scalars;\n";

    // Body: walk the forest roots (unreferenced nodes; children execute
    // through their parents — the walker's root discipline).
    for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
        bool referenced = false;
        for (const KernelNode& n : kernel.nodes) {
            for (const uint32_t c : n.children) {
                referenced = referenced || c == i;
            }
        }
        if (referenced) continue;
        const KernelNode& top = kernel.nodes[i];
        if (em.multiDim) {
            auto r = em.emitNode(i);
            if (!r.has_value()) {
                return std::unexpected<Error>(r.error());
            }
        } else {
            // Legacy 1-D form: exactly the 1-D executor's top-level
            // discipline (loop-driven elementwise; everything else is
            // recorded or rejected honestly).
            switch (top.op) {  // Rule 78: exhaustive
                case KernelOp::Loop: {
                    std::string endCmp;
                    if (top.end == constants::kKernelLoopDynamicBound) {
                        endCmp = "n";
                        em.legacyUsesDynamicN = true;
                    } else {
                        endCmp = i64(top.end);
                    }
                    const std::string var =
                        em.uniqueName(symbols.text(top.var));
                    em.varNames.push_back(var);
                    em.body += "    for (int64_t " + var + " = " +
                               i64(top.begin) + "; " + var + " < " +
                               endCmp + "; ++" + var + ") {\n";
                    em.depth = 1;
                    MLK_TRYV(em.emitChildList(top.children, true));
                    em.depth = 0;
                    em.body += "    }\n";
                    em.varNames.pop_back();
                    break;
                }
                case KernelOp::Call:
                    return err(ErrorCode::UnsupportedCapability,
                               "emitter: Call nodes must be lowered "
                               "before emission (poly.synth)",
                               121);
                case KernelOp::Guard:
                    return err(ErrorCode::UnsupportedCapability,
                               "emitter: speculative guards belong to "
                               "ExecutionEngine (Rule 5)");
                case KernelOp::AllocBuffer:
                case KernelOp::CopyBuffer:
                    return err(ErrorCode::UnsupportedCapability,
                               "emitter: buffer management nodes are "
                               "outside the lowering/polyhedral emission "
                               "alphabet");
                case KernelOp::Compute:
                case KernelOp::Store:
                case KernelOp::Load:
                case KernelOp::Barrier:
                case KernelOp::Trace:
                    // "Standalone occurrences are ignored by this
                    // executor on purpose" — mirrored as recorded
                    // comments, not silent drops.
                    em.body += "    // top-level " +
                               std::string(kernelOpName(top.op)) +
                               " (no loop context: no semantics)\n";
                    break;
                case KernelOp::kCount:
                    return err(ErrorCode::InvalidGraph,
                               "emitter: kCount sentinel");
            }
        }
    }

    if (em.multiDim) {
        em.body += "    return 0;\n}\n";
    } else {
        if (!em.legacyUsesDynamicN) {
            em.body += "    (void)n;\n";
        }
        em.body += "}\n";
    }

    // Prologue assembly: header comments (with the recorded feature
    // counts — Rule 148), includes, optional family snapshot.
    std::string out;
    const std::string kernelName =
        kernel.name != kInvalidSymbolId
            ? symbols.text(kernel.name)
            : std::string("<anonymous>");
    out += "// Generated by MLK+ (mlk_backend_cpu::emitCppSource).\n";
    out += "// Kernel: " + kernelName +
           "  hash: " + std::to_string(kernel.hash()) + "\n";
    out += "// Semantics mirror the buffer executor "
           "(runtime/src/execution/kernel_buffers.cpp); schedule marks "
           "are proven by the polyhedral scheduler "
           "(docs/polyhedral_spec.md #backend).\n";
    out += "// Rule 148 recorded: parallel loops=" +
           std::to_string(em.parallelLoops) + ", simd loops=" +
           std::to_string(em.simdLoops) + ", family=" +
           (em.needsPoly7 ? "poly7" : "libm") + "; pragmas are guarded "
           "by #ifdef _OPENMP (inert without -fopenmp; parallel rows "
           "write disjoint slabs, so any schedule is deterministic).\n";
    out += "#include <cmath>\n#include <cstdint>\n";
    if (em.needsPoly7) emitPoly7Prologue(out);
    out += "\n";
    out += em.body;

    return out;
}

}  // namespace mlk
