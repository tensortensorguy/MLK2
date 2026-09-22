// x86-64 assembly emitter implementation (see header for the ABI/target
// contract). Semantics mirror runtime/src/execution/kernel_buffers.cpp
// EXACTLY — the MultiDimWalker and the 1-D executor — so a differential
// test between the buffer walker and the assembled artifact is bit-exact
// by construction (tests/unit/unit_poly.cpp asm_backend_* tests).
//
// Layout discipline (documented, deterministic):
//   - loop induction vars live in STACK SLOTS at -8*(d+1)(%rbp), d =
//     loop depth outermost-first (uniform addressing over unbounded nest
//     depth; the registers stay free for address arithmetic),
//   - fused-chain temps live at -8*(maxDepth+1+k)(%rbp), k = temp index,
//   - one scratch double slot sits at -8*(maxDepth+maxTemps+1)(%rbp)
//     (values that must survive a libm PLT call — every xmm register is
//     caller-saved in the System V ABI),
//   - all immediate loads go through checked forms: movq with an imm32
//     sign-extended operand, or movabsq for the full int64 range.
#include "mlk/backend/asm_emitter.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <string>

#include "mlk/backend/cpp_emitter.h"
#include "mlk/core/constants.h"
#include "mlk/support/math_families.h"

namespace mlk {

namespace {

/// Same temp-slot cap as the executor and the C++ emitter (Rule 27).
inline constexpr std::size_t kAsmMaxTempSlots = 64;

[[nodiscard]] bool isUnaryOp(MathOp op) noexcept {
    return op == MathOp::Neg || op == MathOp::Exp || op == MathOp::Log ||
           op == MathOp::Sin || op == MathOp::Cos || op == MathOp::Tan ||
           op == MathOp::Tanh || op == MathOp::Sqrt ||
           op == MathOp::Rsqrt || op == MathOp::Erf || op == MathOp::Gelu;
}

[[nodiscard]] std::string i64(const int64_t v) { return std::to_string(v); }

/// Imm32-operands are sign-extended by movq; anything wider needs
/// movabsq (the full 64-bit immediate form).
[[nodiscard]] bool fitsImm32(const int64_t v) noexcept {
    return v >= -2147483648 && v <= 2147483647;
}

[[nodiscard]] int64_t coeffAt(const SmallVector<int64_t, 4>& coeffs,
                              const int depth) noexcept {
    return depth >= 0 &&
                   static_cast<std::size_t>(depth) < coeffs.size()
               ? coeffs[static_cast<std::size_t>(depth)]
               : 0;
}

struct AsmEmitter {
    const KernelModule& km;
    SymbolTable& symbols;
    bool multiDim{false};
    bool needsPoly7{false};  // a Sin site applies the verified poly7 family
    std::size_t parallelLoops{0};
    std::size_t simdLoops{0};
    // Packed-2 execution (see emitPackedLoop): innermost loops whose body
    // qualifies run two iterations per pass with SSE2 pair ops. Each lane
    // performs EXACTLY the scalar iteration's IEEE double operations in
    // the original per-cell order, so results stay bit-exact by
    // construction; odd trip counts take the scalar remainder loop.
    std::vector<bool> packable{};  // per node id (loops only)
    bool packingEnabled{false};    // frame supports pair temps + vec slots
    std::size_t packedLoops{0};
    int frameTemps{0};             // temp-slot region size (slots)
    int vecSlots{0};               // 0 or 2: even-end + saved end slots
    std::string text{};    // .text section (prologue comments + function)
    std::string rodata{};  // .rodata literal pool
    std::size_t litCount{0};
    std::size_t labelCount{0};
    int depth{0};       // current loop depth (0 = module top level)
    int maxDepth{0};    // pre-scan: deepest loop nest
    int maxTemps{0};    // pre-scan: longest fused chain
    std::vector<bool> bindable{};     // per buffer id
    std::vector<int> ptrArgIdx{};     // per buffer id: arg slot of ptr
    int totalArgs{0};

    [[nodiscard]] std::string newLabel(const char* base) {
        return std::string(".") + base + i64(static_cast<int64_t>(
                                                 labelCount++));
    }

    /// Var slot address for loop depth d (d < depth while emitting).
    [[nodiscard]] std::string varSlot(const int d) const {
        return i64(-8 * (d + 1)) + "(%rbp)";
    }

    /// Temp slot address for fused-chain index k. In packed bodies temp
    /// k occupies the PAIR (2k, 2k+1) — callers pass pre-doubled indices;
    /// scalar bodies keep the dense k indexing (same region, no overlap:
    /// a packed main loop finishes before any scalar tail runs).
    [[nodiscard]] std::string tempSlot(const int k) const {
        return i64(-8 * (maxDepth + 1 + k)) + "(%rbp)";
    }

    /// Scratch double slot (survives PLT calls; all xmm are volatile).
    [[nodiscard]] std::string scratchSlot() const {
        return i64(-8 * (maxDepth + frameTemps + 1)) + "(%rbp)";
    }

    /// Packed-loop reserved slots (when vecSlots == 2): slot 0 holds the
    /// main loop's exclusive even end, slot 1 the saved inclusive end for
    /// the scalar remainder loop. Vectorized loops never nest (only
    /// innermost loops pack; subtrees run sequentially), so two shared
    /// slots serve every packed loop in the module.
    [[nodiscard]] std::string vecSlot(const int k) const {
        return i64(-8 * (maxDepth + frameTemps + 1 + 1 + k)) + "(%rbp)";
    }

    /// Byte displacement for a disp32 memory operand; larger forms are
    /// rejected (dims indices and loop counters never come close; an
    /// honest error beats a silent mis-address).
    [[nodiscard]] static Result<int64_t> checkDisp(const int64_t bytes) {
        if (!fitsImm32(bytes)) {
            return err(ErrorCode::UnsupportedCapability,
                       "asm emitter: displacement exceeds disp32");
        }
        return bytes;
    }

    /// Loads argument k (0-based, System V AMD64) into a GP register.
    /// Every argument lives in a HOME STACK SLOT spilled by the
    /// prologue: argument registers are caller-saved, and a libm PLT
    /// call inside a compute (sin/exp/tanh/poly7 helpers) clobbers
    /// %rdi..%r9 — reading an argument register after such a call would
    /// silently use a clobbered pointer. Home slots survive every call.
    void emitLoadArg(const int k, const char* reg) {
        text += "movq " + argSlot(k) + ", " + reg + "\n";
    }

    /// Home slot of argument k (above the var/temp/scratch slots).
    [[nodiscard]] std::string argSlot(const int k) const {
        return i64(-8 * (maxDepth + frameTemps + vecSlots + 2 + k)) +
               "(%rbp)";
    }

    /// Loads an int64 immediate into %rax.
    void emitLoadI64(const int64_t v) {
        if (fitsImm32(v)) {
            text += "movq $" + i64(v) + ", %rax\n";
        } else {
            text += "movabsq $" + i64(v) + ", %rax\n";
        }
    }

    /// Walker flatIndex contract: flat = offset + sum coeffs[p]*var[p]
    /// over min(coeffs, vars) — zero coefficients never bind a var (the
    /// walker skips them before its unbound-var check). Result in %rax.
    /// Signed wrap on imul/add mirrors the executor's int64 arithmetic;
    /// the scheduler proves the realized forms in-range (documented in
    /// docs/polyhedral_spec.md §backend).
    [[nodiscard]] Result<void> emitAffineToRax(
        const SmallVector<int64_t, 4>& coeffs, const int64_t offset) {
        emitLoadI64(offset);
        for (std::size_t p = 0; p < coeffs.size(); ++p) {
            const int64_t c = coeffs[p];
            if (c == 0) continue;
            if (p >= static_cast<std::size_t>(depth)) {
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: affine form references an "
                           "unbound loop var");
            }
            text += "movq " + varSlot(static_cast<int>(p)) + ", %rcx\n";
            if (c == 1) {
                text += "addq %rcx, %rax\n";
            } else if (c == -1) {
                text += "subq %rcx, %rax\n";
            } else if (fitsImm32(c)) {
                text += "imulq $" + i64(c) + ", %rcx, %rcx\n";
                text += "addq %rcx, %rax\n";
            } else {
                text += "movabsq $" + i64(c) + ", %rdx\n";
                text += "imulq %rdx, %rcx, %rcx\n";
                text += "addq %rcx, %rax\n";
            }
        }
        return {};
    }

    /// Flat byte address of an ElemIdx form in %rax with the buffer base
    /// pointer in %r10 (row-major dense: byte address = flat * 8).
    [[nodiscard]] Result<void> emitElemAddress(const int64_t bufBid) {
        if (bufBid < 0 ||
            static_cast<std::size_t>(bufBid) >= km.buffers.size()) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: buffer id out of range");
        }
        if (!bindable[static_cast<std::size_t>(bufBid)]) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: reference to an unbindable buffer");
        }
        return {};
    }

    /// Loads one scalar operand into the TARGET xmm register — %xmm0
    /// for the left operand, %xmm1 for the right (KernelExpr is flat:
    /// at most two live operand values; temps are materialized in stack
    /// slots). The owning Compute node provides the legacy ElemA/ElemB
    /// buffers (walker evalOne contract).
    [[nodiscard]] Result<void> emitOperand(const KernelOperand& o,
                                           const KernelNode& compute,
                                           const char* xmm) {
        switch (o.kind) {  // Rule 78: exhaustive
            case KernelOperand::Kind::Const: {
                const auto bits =
                    std::bit_cast<uint64_t>(o.constValue);
                if (bits == 0) {
                    text += std::string("pxor ") + xmm + ", " + xmm +
                            "\n";  // exact +0.0
                } else {
                    text += "movabsq $" + i64(static_cast<int64_t>(bits)) +
                            ", %rax\nmovq %rax, " + xmm + "\n";
                }
                return {};
            }
            case KernelOperand::Kind::ElemA:
            case KernelOperand::Kind::ElemB: {
                // Multi-dim computes reject legacy operands exactly like
                // the walker's execPair; in the legacy 1-D form they are
                // buffer[i] with i the single enclosing loop var.
                if (multiDim) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: legacy element operand in a "
                               "multi-dim compute");
                }
                if (depth == 0) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: legacy operand outside a "
                               "loop");
                }
                const uint32_t bid =
                    o.kind == KernelOperand::Kind::ElemA
                        ? compute.bufferA
                        : compute.bufferB;
                auto ok = emitElemAddress(static_cast<int64_t>(bid));
                if (!ok.has_value()) {
                    return std::unexpected<Error>(ok.error());
                }
                emitLoadArg(ptrArgIdx[bid], "%r10");
                text += "movq " + varSlot(depth - 1) + ", %rax\n";
                text += "salq $3, %rax\n";
                text += "movsd (%r10,%rax,1), " + std::string(xmm) +
                        "\n";
                return {};
            }
            case KernelOperand::Kind::ScalarParam: {
                // Walker: idx < scalars.size() ? scalars[idx] : 0.0 —
                // the artifact mirrors the C++ emitter's signed compare
                // (a negative count yields 0.0, never a wild read).
                const int64_t idx = o.index;
                if (idx < 0) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: negative scalar index");
                }
                const std::string oob = newLabel("Lscal_oob");
                const std::string done = newLabel("Lscal_end");
                // n_scalars is the last argument in BOTH ABI forms; the
                // scalars pointer precedes it (signature pre-scan in
                // emitAsmSource fixed totalArgs before body emission).
                const int countArg = totalArgs - 1;
                emitLoadArg(countArg, "%rax");
                text += "cmpq $" + i64(idx) + ", %rax\n";
                text += "jle " + oob + "\n";
                emitLoadArg(countArg - 1, "%r10");
                if (idx <= 2147483647 / 8) {
                    text += "movsd " + i64(idx * 8) + "(%r10), " +
                            std::string(xmm) + "\n";
                } else {
                    text += "movabsq $" + i64(idx) + ", %rax\n";
                    text += "salq $3, %rax\n";
                    text += "movsd (%r10,%rax,1), " + std::string(xmm) +
                            "\n";
                }
                text += "jmp " + done + "\n" + oob + ":\npxor " +
                        std::string(xmm) + ", " + xmm + "\n" + done +
                        ":\n";
                return {};
            }
            case KernelOperand::Kind::Temp: {
                if (o.index < 0 ||
                    static_cast<std::size_t>(o.index) >=
                        kAsmMaxTempSlots) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: temp index out of range");
                }
                text += "movq " + tempSlot(static_cast<int>(o.index)) +
                        ", %rax\nmovq %rax, " + std::string(xmm) + "\n";
                return {};
            }
            case KernelOperand::Kind::ElemIdx: {
                auto ok = emitElemAddress(o.index);
                if (!ok.has_value()) {
                    return std::unexpected<Error>(ok.error());
                }
                auto flat =
                    emitAffineToRax(o.idxCoeffs, o.idxOffset);
                if (!flat.has_value()) {
                    return std::unexpected<Error>(flat.error());
                }
                emitLoadArg(ptrArgIdx[static_cast<std::size_t>(o.index)],
                            "%r10");
                text += "salq $3, %rax\n";
                text += "movsd (%r10,%rax,1), " + std::string(xmm) + "\n";
                return {};
            }
        }
        return err(ErrorCode::InvalidGraph,
                   "asm emitter: unknown operand kind");
    }

    /// One rodata double literal; returns the RIP-relative operand.
    [[nodiscard]] std::string rodataDouble(const double d) {
        const auto bits = std::bit_cast<uint64_t>(d);
        const std::string label = ".LC" + i64(static_cast<int64_t>(
                                                litCount++));
        rodata += label + ":\n.quad 0x" +
                  [&] {
                      char buf[32];
                      std::snprintf(buf, sizeof(buf), "%016llx",
                                    static_cast<unsigned long long>(
                                        bits));
                      return std::string(buf);
                  }() +
                  "\n";
        return label + "(%rip)";
    }

    [[nodiscard]] Result<void> emitLibmCall(const char* fn) {
        text += std::string("call ") + fn + "@PLT\n";
        return {};
    }

    /// One scalar op application — result in %xmm0 (walker apply()
    /// contract verbatim, including the singularity-as-value special
    /// cases and the Pow(2,2) fast form; Rule 93: mathematical
    /// exceptions are VALUES, not errors). Operation ORDER matches the
    /// C expressions the walker and the C++ emitter evaluate, so every
    /// rounding step is identical (Rules 33/90).
    [[nodiscard]] Result<void> emitApply(const KernelExpr& e,
                                         const KernelNode& compute) {
        auto a = emitOperand(e.a, compute, "%xmm0");
        if (!a.has_value()) return std::unexpected<Error>(a.error());
        const bool unary = isUnaryOp(e.op);
        if (!unary) {
            auto bv = emitOperand(e.b, compute, "%xmm1");
            if (!bv.has_value()) return std::unexpected<Error>(bv.error());
        }
        // Family dispatch: the walker applies "poly7" to Sin only
        // (verified family, math_families.h certificate). The artifact
        // calls the local helper below — the SAME reduction/polynomial
        // operation order as the C reference, so the rounding steps are
        // identical and the differential stays bit-exact (Rules 33/90).
        if (e.op == MathOp::Sin &&
            compute.family != kInvalidSymbolId &&
            symbols.text(compute.family) == "poly7") {
            needsPoly7 = true;
            text += "call mlk_poly_sin\n";
            return {};
        }
        switch (e.op) {  // Rule 78: exhaustive over scalar-realizable ops
            case MathOp::Add:
                text += "addsd %xmm1, %xmm0\n";
                return {};
            case MathOp::Sub:
                text += "subsd %xmm1, %xmm0\n";
                return {};
            case MathOp::Mul:
                text += "mulsd %xmm1, %xmm0\n";
                return {};
            case MathOp::Div: {
                // b != 0.0 ? a / b : 0.0 — NaN compares unordered (so a
                // NaN divisor DIVIDES, exactly like the C condition).
                const std::string l0 = newLabel("Lop");
                const std::string l1 = newLabel("Lop");
                text += "pxor %xmm2, %xmm2\n";
                text += "ucomisd %xmm2, %xmm1\n";
                text += "jp " + l0 + "\n";
                text += "jne " + l0 + "\n";
                text += "pxor %xmm0, %xmm0\n";
                text += "jmp " + l1 + "\n" + l0 + ":\n";
                text += "divsd %xmm1, %xmm0\n" + l1 + ":\n";
                return {};
            }
            case MathOp::Neg:
                // C unary minus: sign-bit flip (exact for -0.0/NaN).
                text += "movabsq $-9223372036854775808, %rax\n";
                text += "movq %rax, %xmm1\n";
                text += "xorpd %xmm1, %xmm0\n";
                return {};
            case MathOp::Pow: {
                // a == 2.0 && b == 2.0 ? a * a : pow(a, b)
                const std::string fast = newLabel("Lop");
                const std::string done = newLabel("Lop");
                text += "ucomisd " + rodataDouble(2.0) + ", %xmm0\n";
                text += "jp " + fast + "\njne " + fast + "\n";
                text += "ucomisd " + rodataDouble(2.0) + ", %xmm1\n";
                text += "jp " + fast + "\njne " + fast + "\n";
                text += "mulsd %xmm0, %xmm0\njmp " + done + "\n";
                text += fast + ":\n";
                auto r = emitLibmCall("pow");
                if (!r.has_value()) return r;
                text += done + ":\n";
                return {};
            }
            case MathOp::Exp:
                return emitLibmCall("exp");
            case MathOp::Log: {
                // a > 0.0 ? log(a) : 0.0 (unordered -> 0.0)
                const std::string l0 = newLabel("Lop");
                const std::string l1 = newLabel("Lop");
                text += "pxor %xmm2, %xmm2\n";
                text += "ucomisd %xmm2, %xmm0\n";
                text += "ja " + l0 + "\n";
                text += "pxor %xmm0, %xmm0\njmp " + l1 + "\n" + l0 +
                        ":\n";
                {
                    auto r = emitLibmCall("log");
                    if (!r.has_value()) return r;
                }
                text += l1 + ":\n";
                return {};
            }
            case MathOp::Sin:
                return emitLibmCall("sin");
            case MathOp::Cos:
                return emitLibmCall("cos");
            case MathOp::Tan:
                return emitLibmCall("tan");
            case MathOp::Tanh:
                return emitLibmCall("tanh");
            case MathOp::Sqrt: {
                // a >= 0.0 ? sqrt(a) : 0.0; sqrtsd is correctly rounded
                // and identical to libm sqrt (and to what cc emits).
                const std::string l0 = newLabel("Lop");
                const std::string l1 = newLabel("Lop");
                text += "pxor %xmm2, %xmm2\n";
                text += "ucomisd %xmm2, %xmm0\n";
                text += "jae " + l0 + "\n";
                text += "pxor %xmm0, %xmm0\njmp " + l1 + "\n" + l0 +
                        ":\n";
                text += "sqrtsd %xmm0, %xmm0\n" + l1 + ":\n";
                return {};
            }
            case MathOp::Rsqrt: {
                // a > 0.0 ? 1.0 / sqrt(a) : 0.0
                const std::string l0 = newLabel("Lop");
                const std::string l1 = newLabel("Lop");
                text += "pxor %xmm2, %xmm2\n";
                text += "ucomisd %xmm2, %xmm0\n";
                text += "ja " + l0 + "\n";
                text += "pxor %xmm0, %xmm0\njmp " + l1 + "\n" + l0 +
                        ":\n";
                text += "sqrtsd %xmm0, %xmm0\n";
                text += "movsd " + rodataDouble(1.0) + ", %xmm1\n";
                text += "divsd %xmm0, %xmm1\n";
                text += "movaps %xmm1, %xmm0\n" + l1 + ":\n";
                return {};
            }
            case MathOp::Erf:
                return emitLibmCall("erf");
            case MathOp::Gelu: {
                // 0.5 * a * (1.0 + tanh(0.7978845608028654 *
                //                        (a + 0.044715 * a * a * a)))
                // — left-associative exactly as written; `a` lives in a
                // stack scratch across the tanh PLT call (all xmm
                // registers are caller-saved).
                text += "movsd %xmm0, " + scratchSlot() + "\n";
                text += "movsd " + rodataDouble(0.044715) + ", %xmm1\n";
                text += "mulsd %xmm0, %xmm1\n";  // 0.044715*a
                text += "mulsd %xmm0, %xmm1\n";  // *a
                text += "mulsd %xmm0, %xmm1\n";  // *a
                text += "addsd %xmm0, %xmm1\n";  // a + m3
                text += "movsd " +
                        rodataDouble(0.7978845608028654) + ", %xmm2\n";
                text += "mulsd %xmm1, %xmm2\n";  // 0.797...*s1
                text += "movaps %xmm2, %xmm0\n";
                {
                    auto r = emitLibmCall("tanh");
                    if (!r.has_value()) return r;
                }
                text += "movsd " + rodataDouble(1.0) + ", %xmm1\n";
                text += "addsd %xmm1, %xmm0\n";  // 1.0 + tanh
                text += "movaps %xmm0, %xmm2\n";
                text += "movsd " + scratchSlot() + ", %xmm1\n";  // a
                text += "movsd " + rodataDouble(0.5) + ", %xmm0\n";
                text += "mulsd %xmm1, %xmm0\n";  // 0.5*a
                text += "mulsd %xmm2, %xmm0\n";  // * (1 + tanh)
                return {};
            }
            default:
                // Non-realizable ops are rejected at lowering time
                // (isScalarRealizable); reaching here is a contract bug.
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: non-realizable math op");
        }
    }

    /// Emits a fused Compute chain + its Store (walker execPair / 1-D
    /// runElementwiseRange contract). Multi-dim stores carry the
    /// negative-flat check the walker performs (InvalidGraph there, ABI
    /// code 1 here — the driver maps it back).
    [[nodiscard]] Result<void> emitComputePair(const KernelNode& compute,
                                               const KernelNode& store) {
        if (compute.exprs.size() > kAsmMaxTempSlots) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: compute chain exceeds the temp-slot "
                       "cap");
        }
        if (multiDim) {
            for (std::size_t k = 0; k < compute.exprs.size(); ++k) {
                const KernelExpr& e = compute.exprs[k];
                const auto badTemp = [&](const KernelOperand& o) {
                    return o.kind == KernelOperand::Kind::Temp &&
                           o.index >= static_cast<int64_t>(k);
                };
                if (badTemp(e.a) || badTemp(e.b)) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: temp operand is not earlier "
                               "in the chain");
                }
                auto r = emitApply(e, compute);
                if (!r.has_value()) return r;
                text += "movsd %xmm0, " +
                        tempSlot(static_cast<int>(k)) + "\n";
            }
            // Chain value: last temp (or 0.0 for the documented
            // degenerate empty-chain shape, exactly like the walker).
            if (compute.exprs.empty()) {
                text += "pxor %xmm0, %xmm0\n";
            } else {
                text += "movq " +
                        tempSlot(static_cast<int>(
                            compute.exprs.size() - 1)) +
                        ", %rax\nmovq %rax, %xmm0\n";
            }
            // Store target: flat * 8 with the base pointer in %r10.
            auto ok =
                emitElemAddress(static_cast<int64_t>(store.bufferOut));
            if (!ok.has_value()) {
                return std::unexpected<Error>(ok.error());
            }
            auto flat = emitAffineToRax(store.outIndexCoeffs,
                                        store.outIndexOffset);
            if (!flat.has_value()) {
                return std::unexpected<Error>(flat.error());
            }
            emitLoadArg(ptrArgIdx[store.bufferOut], "%r10");
            text += "salq $3, %rax\n";
            text += "testq %rax, %rax\n";
            text += "js .Lmlk_neg_store\n";
            if (store.accum == AccumMode::Add) {
                text += "movsd (%r10,%rax,1), %xmm1\n";
                text += "addsd %xmm0, %xmm1\n";
                text += "movsd %xmm1, (%r10,%rax,1)\n";
            } else if (store.accum == AccumMode::Max) {
                // Order-insensitive row-max primitive — the EXACT
                // walker select (execPair): write the value only when
                // value > cur (ordered). comisd value,cur sets CF=1|ZF=1
                // for <=, ties, AND unordered (NaN — PF/ZF/CF all set),
                // so jbe keeps the running slot on NaN and +/-0 ties
                // exactly like (value > cur) ? value : cur; vmaxsd
                // would break the NaN rule.
                const std::string keep = newLabel("Lmaxkeep");
                text += "movsd (%r10,%rax,1), %xmm1\n";
                text += "comisd %xmm1, %xmm0\n";
                text += "jbe " + keep + "\n";
                text += "movsd %xmm0, (%r10,%rax,1)\n";
                text += keep + ":\n";
            } else {
                text += "movsd %xmm0, (%r10,%rax,1)\n";
            }
            return {};
        }

        // Legacy 1-D form: out[i] = value (i = the innermost loop var).
        if (compute.exprs.empty()) {
            // Legacy single-op compute: one op over ElemA (+ElemB for
            // binary ops) — the 1-D fast path.
            KernelExpr single;
            single.op = compute.math;
            single.a.kind = KernelOperand::Kind::ElemA;
            single.b.kind = KernelOperand::Kind::ElemB;
            auto r = emitApply(single, compute);
            if (!r.has_value()) return r;
        } else {
            for (std::size_t k = 0; k < compute.exprs.size(); ++k) {
                const KernelExpr& e = compute.exprs[k];
                const auto badTemp = [&](const KernelOperand& o) {
                    return o.kind == KernelOperand::Kind::Temp &&
                           o.index >= static_cast<int64_t>(k);
                };
                if (badTemp(e.a) || badTemp(e.b)) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: temp operand is not earlier "
                               "in the chain");
                }
                auto r = emitApply(e, compute);
                if (!r.has_value()) return r;
                text += "movsd %xmm0, " +
                        tempSlot(static_cast<int>(k)) + "\n";
            }
            text += "movq " +
                    tempSlot(static_cast<int>(compute.exprs.size() - 1)) +
                    ", %rax\nmovq %rax, %xmm0\n";
        }
        auto ok = emitElemAddress(static_cast<int64_t>(store.bufferOut));
        if (!ok.has_value()) {
            return std::unexpected<Error>(ok.error());
        }
        if (depth == 0) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: store outside a loop");
        }
        emitLoadArg(ptrArgIdx[store.bufferOut], "%r10");
        text += "movq " + varSlot(depth - 1) + ", %rax\n";
        text += "salq $3, %rax\n";
        if (store.accum == AccumMode::Add) {
            text += "movsd (%r10,%rax,1), %xmm1\n";
            text += "addsd %xmm0, %xmm1\n";
            text += "movsd %xmm1, (%r10,%rax,1)\n";
        } else if (store.accum == AccumMode::Max) {
            // The 1-D executor routes every Max module through the
            // multi-dim walker; a legacy-form Max never reaches this
            // emitter (kept as an honest structural rejection).
            return err(ErrorCode::UnsupportedCapability,
                       "asm emitter: max-accumulate stores belong to "
                       "the multi-dim form (walker parity)");
        } else {
            text += "movsd %xmm0, (%r10,%rax,1)\n";
        }
        return {};
    }

    /// Loads one PACKED (2-lane) operand into xmm. Lane 0 is iteration v,
    /// lane 1 is iteration v+1 of the packed loop (var slot varSlot(d)):
    ///   coeff of v == +1 -> the two iterations read ADJACENT cells: one
    ///                       movupd fetches both (flat form of v);
    ///   coeff of v == 0  -> both iterations read the SAME cell: one
    ///                       scalar load, duplicated with unpcklpd;
    ///   Const            -> one scalar literal, duplicated (or pxor for
    ///                       exact +0.0 in both lanes);
    ///   Temp index t     -> the chain's pair temp (slots 2t, 2t+1).
    [[nodiscard]] Result<void> emitOperandPacked(const KernelOperand& o,
                                                 const int varDepth,
                                                 const char* xmm) {
        switch (o.kind) {  // Rule 78: exhaustive
            case KernelOperand::Kind::Const: {
                const auto bits = std::bit_cast<uint64_t>(o.constValue);
                if (bits == 0) {
                    text += std::string("pxor ") + xmm + ", " + xmm +
                            "\n";  // exact +0.0 in BOTH lanes
                } else {
                    text += "movabsq $" + i64(static_cast<int64_t>(bits)) +
                            ", %rax\nmovq %rax, " + xmm + "\n";
                    text += std::string("unpcklpd ") + xmm + ", " + xmm +
                            "\n";
                }
                return {};
            }
            case KernelOperand::Kind::Temp: {
                if (o.index < 0 ||
                    static_cast<std::size_t>(o.index) >=
                        kAsmMaxTempSlots / 2) {
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: temp index out of range");
                }
                // The slot layout DESCENDS with the index (slot k+1 sits
                // 8 bytes below slot k), so the 16-byte pair access must
                // start at the LOWER address — tempSlot(2t+1) — covering
                // exactly the (2t, 2t+1) pair without touching the var
                // region above it.
                text += "movupd " +
                        tempSlot(2 * static_cast<int>(o.index) + 1) +
                        ", " + std::string(xmm) + "\n";
                return {};
            }
            case KernelOperand::Kind::ElemIdx: {
                auto ok = emitElemAddress(o.index);
                if (!ok.has_value()) {
                    return std::unexpected<Error>(ok.error());
                }
                auto flat = emitAffineToRax(o.idxCoeffs, o.idxOffset);
                if (!flat.has_value()) {
                    return std::unexpected<Error>(flat.error());
                }
                emitLoadArg(ptrArgIdx[static_cast<std::size_t>(o.index)],
                            "%r10");
                text += "salq $3, %rax\n";
                if (coeffAt(o.idxCoeffs, varDepth) == 1) {
                    text += std::string("movupd (%r10,%rax,1), ") + xmm +
                            "\n";
                } else {
                    text += "movsd (%r10,%rax,1), " + std::string(xmm) +
                            "\n";
                    text += std::string("unpcklpd ") + xmm + ", " + xmm +
                            "\n";
                }
                return {};
            }
            case KernelOperand::Kind::ElemA:
            case KernelOperand::Kind::ElemB:
            case KernelOperand::Kind::ScalarParam:
                // Excluded by analyzePackedLoop; reaching here is a
                // pre-pass contract bug — fail honestly, never emit a
                // scalar operand into a pair context.
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: non-packable operand in a "
                           "packed body");
        }
        return err(ErrorCode::InvalidGraph,
                   "asm emitter: unknown operand kind");
    }

    /// One packed op application — pair result in %xmm0. Only the
    /// elementwise associative-free forms with an exact per-lane match to
    /// the scalar op are packable (analyzePackedLoop gates this): each
    /// lane performs the SAME IEEE double operation the scalar iteration
    /// performs, in the same order (Rules 33/90).
    [[nodiscard]] Result<void> emitApplyPacked(const KernelExpr& e) {
        auto a = emitOperandPacked(e.a, depth - 1, "%xmm0");
        if (!a.has_value()) return std::unexpected<Error>(a.error());
        auto bv = emitOperandPacked(e.b, depth - 1, "%xmm1");
        if (!bv.has_value()) return std::unexpected<Error>(bv.error());
        switch (e.op) {  // gated to Add/Sub/Mul by analyzePackedLoop
            case MathOp::Add:
                text += "addpd %xmm1, %xmm0\n";
                return {};
            case MathOp::Sub:
                text += "subpd %xmm1, %xmm0\n";
                return {};
            case MathOp::Mul:
                text += "mulpd %xmm1, %xmm0\n";
                return {};
            default:
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: non-packable math op");
        }
    }

    /// A Compute+Store pair in packed form (the pair twin of
    /// emitComputePair's multi-dim branch). The chain runs on pair
    /// values (temp k = slots 2k, 2k+1); the store is one movupd RMW /
    /// overwrite of the two adjacent cells the two iterations write.
    [[nodiscard]] Result<void> emitComputePairPacked(
        const KernelNode& compute, const KernelNode& store) {
        if (compute.exprs.size() > kAsmMaxTempSlots / 2) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: packed chain exceeds the temp-slot "
                       "cap");
        }
        for (std::size_t k = 0; k < compute.exprs.size(); ++k) {
            const KernelExpr& e = compute.exprs[k];
            const auto badTemp = [&](const KernelOperand& o) {
                return o.kind == KernelOperand::Kind::Temp &&
                       o.index >= static_cast<int64_t>(k);
            };
            if (badTemp(e.a) || badTemp(e.b)) {
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: temp operand is not earlier in "
                           "the chain");
            }
            auto r = emitApplyPacked(e);
            if (!r.has_value()) return r;
            // Pair spill at the LOWER address of the (2k, 2k+1) pair —
            // see the slot-layout note in emitOperandPacked.
            text += "movupd %xmm0, " +
                    tempSlot(2 * static_cast<int>(k) + 1) + "\n";
        }
        // Chain value: last temp pair (or the +0.0 pair for the
        // documented degenerate empty-chain shape, exactly like the
        // walker's scalar path).
        if (compute.exprs.empty()) {
            text += "pxor %xmm0, %xmm0\n";
        } else {
            text += "movupd " +
                    tempSlot(2 * static_cast<int>(
                                 compute.exprs.size() - 1) +
                             1) +
                    ", %xmm0\n";
        }
        auto ok = emitElemAddress(static_cast<int64_t>(store.bufferOut));
        if (!ok.has_value()) {
            return std::unexpected<Error>(ok.error());
        }
        auto flat = emitAffineToRax(store.outIndexCoeffs,
                                    store.outIndexOffset);
        if (!flat.has_value()) {
            return std::unexpected<Error>(flat.error());
        }
        emitLoadArg(ptrArgIdx[store.bufferOut], "%r10");
        text += "salq $3, %rax\n";
        text += "testq %rax, %rax\n";
        text += "js .Lmlk_neg_store\n";
        if (store.accum == AccumMode::Add) {
            text += "movupd (%r10,%rax,1), %xmm1\n";
            text += "addpd %xmm0, %xmm1\n";
            text += "movupd %xmm1, (%r10,%rax,1)\n";
        } else {
            text += "movupd %xmm0, (%r10,%rax,1)\n";
        }
        return {};
    }

    /// Emits the innermost loop in PACKED-2 form: a main loop stepping
    /// the induction var by 2 whose body evaluates every Compute/Store
    /// pair on lane pairs (v, v+1), plus a scalar remainder loop running
    /// the ORIGINAL body for the final trip%2 iterations. Both loops
    /// share the var slot; the inclusive end is saved in a vec slot for
    /// the remainder's exit test.
    ///
    /// Bit-exactness: lane 0 replays the even iterations, lane 1 the odd
    /// ones — each with the scalar op sequence — and the remainder loop
    /// is the unmodified scalar body; per-cell operation order is
    /// therefore exactly the scalar order (the gates in
    /// analyzePackedLoop keep every cell's chain self-contained).
    [[nodiscard]] Result<void> emitPackedLoop(const KernelNode& n,
                                             const int varDepth) {
        const std::string top = newLabel("Lpkmain");
        const std::string mainEnd = newLabel("Lpkmainend");
        const std::string tailTop = newLabel("Lpktail");
        const std::string loopEnd = newLabel("Lpkend");
        // begin -> var slot (const | affine over the enclosing stack).
        if (n.beginCoeffs.empty()) {
            emitLoadI64(n.begin);
        } else {
            auto r = emitAffineToRax(n.beginCoeffs, n.beginOffset);
            if (!r.has_value()) return r;
        }
        text += "movq %rax, " + varSlot(varDepth) + "\n";
        // endInclusive -> %rax (existing three-form discipline).
        if (!n.endCoeffs.empty()) {
            auto r = emitAffineToRax(n.endCoeffs, n.endOffset);
            if (!r.has_value()) return r;
        } else if (n.end == constants::kKernelLoopDynamicBound &&
                   n.endBuf != constants::kInvalidId &&
                   n.endBuf < km.buffers.size() && n.endDim >= 0 &&
                   n.endDim < static_cast<int32_t>(
                                  km.buffers[n.endBuf].dims.size())) {
            if (!bindable[n.endBuf]) {
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: bound source buffer is not "
                           "bindable");
            }
            emitLoadArg(ptrArgIdx[n.endBuf], "%r10");
            auto disp =
                checkDisp(static_cast<int64_t>(n.endDim) * 8);
            if (!disp.has_value()) {
                return std::unexpected<Error>(disp.error());
            }
            text += "movq " + i64(*disp) + "(%r10), %rax\n";
            text += "subq $1, %rax\n";
        } else {
            emitLoadI64(n.end - 1);
        }
        // Empty range: the scalar exit test skips both loops.
        text += "cmpq %rax, " + varSlot(varDepth) + "\n";
        text += "jg " + loopEnd + "\n";
        // evenTrip = trip & ~1; evenEndExclusive = v + evenTrip.
        text += "movq %rax, %rcx\n";
        text += "subq " + varSlot(varDepth) + ", %rcx\n";
        text += "incq %rcx\n";
        text += "andq $-2, %rcx\n";
        text += "addq " + varSlot(varDepth) + ", %rcx\n";
        text += "movq %rcx, " + vecSlot(0) + "\n";
        text += "movq %rax, " + vecSlot(1) + "\n";
        // Main packed loop: for (v; v < evenEndExclusive; v += 2).
        text += top + ":\n";
        text += "movq " + vecSlot(0) + ", %rcx\n";
        text += "cmpq %rcx, " + varSlot(varDepth) + "\n";
        text += "jge " + mainEnd + "\n";
        ++depth;
        if (n.parallel) ++parallelLoops;
        if (n.vectorHint) ++simdLoops;
        ++packedLoops;
        for (std::size_t ci = 0; ci < n.children.size(); ++ci) {
            const uint32_t cid = n.children[ci];
            const KernelNode& c = km.nodes[cid];
            if (c.op == KernelOp::Store) continue;  // unconsumed: no-op
            const KernelNode& store = km.nodes[n.children[ci + 1]];
            auto r = emitComputePairPacked(c, store);
            if (!r.has_value()) return r;
            ++ci;  // consume the paired store
        }
        --depth;
        text += "addq $2, " + varSlot(varDepth) + "\n";
        text += "jmp " + top + "\n" + mainEnd + ":\n";
        // Scalar remainder loop: the ORIGINAL body, unmodified.
        text += tailTop + ":\n";
        text += "movq " + vecSlot(1) + ", %rax\n";
        text += "cmpq %rax, " + varSlot(varDepth) + "\n";
        text += "jg " + loopEnd + "\n";
        ++depth;
        auto kids = emitChildList(n.children, false);
        if (!kids.has_value()) {
            return std::unexpected<Error>(kids.error());
        }
        --depth;
        text += "incq " + varSlot(varDepth) + "\n";
        text += "jmp " + tailTop + "\n";
        text += loopEnd + ":\n";
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
                           "asm emitter: child id out of range");
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
                    auto r = emitComputePair(c, *store);
                    if (!r.has_value()) return r;
                } else {
                    // Walker: a bare Compute has no semantics at this
                    // point (needs its paired Store).
                    text += "# bare compute: no paired store (no "
                            "semantics)\n";
                }
                if (store != nullptr && !firstStore) ++ci;  // consume
                continue;
            }
            if (!multiDim) {
                // The 1-D executor inspects loop children for Compute
                // only; everything else carries no semantics there.
                text += std::string("# ") + kernelOpName(c.op) +
                        " (no semantics in the 1-D executor)\n";
                continue;
            }
            auto r = emitNode(cid);
            if (!r.has_value()) return r;
        }
        return {};
    }

    /// One node of the multi-dim forest (walker execNode contract).
    [[nodiscard]] Result<void> emitNode(const uint32_t nodeId) {
        if (nodeId >= km.nodes.size()) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: node id out of range");
        }
        const KernelNode& n = km.nodes[nodeId];
        switch (n.op) {  // Rule 78: exhaustive
            case KernelOp::Loop: {
                if (n.step != 1) {
                    return err(ErrorCode::UnsupportedCapability,
                               "asm emitter: non-unit loop step in a "
                               "multi-dim nest");
                }
                // Packed-2 fast form: qualifying innermost loops run two
                // iterations per pass (see analyzePackedLoop for the
                // legality gates; the header records the packed count).
                if (packingEnabled && nodeId < packable.size() &&
                    packable[nodeId]) {
                    auto r = emitPackedLoop(n, depth);
                    if (!r.has_value()) {
                        return std::unexpected<Error>(r.error());
                    }
                    return {};
                }
                const std::string top = newLabel("Lloop");
                const std::string endl = newLabel("Lloopend");
                // begin = const | affine over the enclosing stack.
                if (n.beginCoeffs.empty()) {
                    emitLoadI64(n.begin);
                } else {
                    auto r = emitAffineToRax(n.beginCoeffs,
                                             n.beginOffset);
                    if (!r.has_value()) return r;
                }
                text += "movq %rax, " + varSlot(depth) + "\n";
                text += top + ":\n";
                // endInclusive = end-1 | affine | dims[endBuf][endDim]-1.
                if (!n.endCoeffs.empty()) {
                    auto r = emitAffineToRax(n.endCoeffs, n.endOffset);
                    if (!r.has_value()) return r;
                } else if (n.end == constants::kKernelLoopDynamicBound &&
                           n.endBuf != constants::kInvalidId &&
                           n.endBuf < km.buffers.size() && n.endDim >= 0 &&
                           n.endDim < static_cast<int32_t>(
                                          km.buffers[n.endBuf]
                                              .dims.size())) {
                    if (!bindable[n.endBuf]) {
                        return err(ErrorCode::InvalidGraph,
                                   "asm emitter: bound source buffer is "
                                   "not bindable");
                    }
                    emitLoadArg(ptrArgIdx[n.endBuf], "%r10");
                    auto disp = checkDisp(
                        static_cast<int64_t>(n.endDim) * 8);
                    if (!disp.has_value()) {
                        return std::unexpected<Error>(disp.error());
                    }
                    text += "movq " + i64(*disp) + "(%r10), %rax\n";
                    text += "subq $1, %rax\n";
                } else if (n.end ==
                           constants::kKernelLoopDynamicBound) {
                    // The walker degrades to io.elements here; a
                    // standalone artifact has no binder (documented
                    // boundary, mirrored from the C++ emitter).
                    return err(ErrorCode::UnsupportedCapability,
                               "asm emitter: legacy dynamic bound in a "
                               "multi-dim module (bind endBuf/endDim or "
                               "constant bounds)");
                } else {
                    emitLoadI64(n.end - 1);
                }
                // for (v = begin; v <= endInclusive; ++v) — exit when
                // v > end (signed); an empty range falls through.
                text += "cmpq %rax, " + varSlot(depth) + "\n";
                text += "jg " + endl + "\n";
                if (n.parallel) ++parallelLoops;
                if (n.vectorHint) ++simdLoops;
                ++depth;
                auto kids = emitChildList(n.children, false);
                if (!kids.has_value()) {
                    return std::unexpected<Error>(kids.error());
                }
                --depth;
                text += "incq " + varSlot(depth) + "\n";
                text += "jmp " + top + "\n" + endl + ":\n";
                return {};
            }
            case KernelOp::Guard: {
                if (!n.hasAffineGuard()) {
                    return err(ErrorCode::UnsupportedCapability,
                               "asm emitter: speculative (non-affine) "
                               "guards belong to ExecutionEngine "
                               "(Rule 5)");
                }
                // CLAST-style affine-equality predicate: children run
                // only where sum coeffs[p]*vars[p] + offset == 0.
                const std::string skip = newLabel("Lskip");
                auto r = emitAffineToRax(n.guardCoeffs, n.guardOffset);
                if (!r.has_value()) return r;
                text += "testq %rax, %rax\n";
                text += "jne " + skip + "\n";
                auto kids = emitChildList(n.children, false);
                if (!kids.has_value()) {
                    return std::unexpected<Error>(kids.error());
                }
                text += skip + ":\n";
                return {};
            }
            case KernelOp::Compute:
                // Paired by emitChildList; a bare compute reaching here
                // has no paired store (walker: no semantics).
                text += "# bare compute: no paired store (no semantics)\n";
                return {};
            case KernelOp::Store:
                // Unconsumed store (walker: no-op).
                text += "# unconsumed store (no-op)\n";
                return {};
            case KernelOp::Load:
                text += "# load (no-op in the artifact)\n";
                return {};
            case KernelOp::Barrier:
                text += "# barrier (no-op: single-region artifact)\n";
                return {};
            case KernelOp::Trace:
                // Rule 96: tracing must remain correct — the runtime
                // hook has no artifact counterpart, so the point is
                // RECORDED, never dropped silently.
                text += "# [mlk:trace] runtime trace point\n";
                return {};
            case KernelOp::Call:
                return err(ErrorCode::UnsupportedCapability,
                           "asm emitter: Call nodes must be lowered "
                           "before emission (poly.synth)",
                           121);
            case KernelOp::AllocBuffer:
            case KernelOp::CopyBuffer:
                return err(ErrorCode::UnsupportedCapability,
                           "asm emitter: buffer management nodes are "
                           "outside the lowering/polyhedral emission "
                           "alphabet");
            case KernelOp::kCount:
                return err(ErrorCode::InvalidGraph,
                           "asm emitter: kCount sentinel");
        }
        return err(ErrorCode::InvalidGraph, "asm emitter: unreachable op");
    }

    /// Pre-scan: deepest loop nest (frame var slots) and longest fused
    /// chain (frame temp slots) over the forest ROOTS (children execute
    /// through their parents — the walker's root discipline).
    Result<void> scanFrame(uint32_t nodeId, int loopDepth) {
        if (nodeId >= km.nodes.size()) {
            return err(ErrorCode::InvalidGraph,
                       "asm emitter: node id out of range");
        }
        const KernelNode& n = km.nodes[nodeId];
        const int childDepth =
            n.op == KernelOp::Loop ? loopDepth + 1 : loopDepth;
        if (childDepth > maxDepth) maxDepth = childDepth;
        const int chain = static_cast<int>(n.exprs.size());
        if (chain > maxTemps) maxTemps = chain;
        for (const uint32_t c : n.children) {
            auto r = scanFrame(c, childDepth);
            if (!r.has_value()) return r;
        }
        return {};
    }

    /// Packed-2 pre-pass: marks the innermost loops whose body can run
    /// two iterations per pass with SSE2 pair ops (emitPackedLoop).
    ///
    /// Legality contract (bit-exact by construction):
    ///   - every lane performs EXACTLY the scalar iteration's double
    ///     operations in the original per-cell order (a packed mul/add is
    ///     the same IEEE op as its scalar twin, one lane per cell);
    ///   - the packed loop's var v must be an elementwise (not reduction)
    ///     dimension of every statement: each Store's flat coefficient of
    ///     v is +1 (the two iterations write ADJACENT DISTINCT cells — a
    ///     v-invariant Add store would double-accumulate and is rejected;
    ///     Max's NaN select has no pair form);
    ///   - every ElemIdx read has coefficient of v in {0, +1}: +1 loads
    ///     the two adjacent cells one `movupd` fetches; 0 reads one cell
    ///     shared by both iterations (identical values — nothing writes
    ///     it in between, enforced by the buffer-disjointness gate);
    ///   - chains use only Add/Sub/Mul over Const / ElemIdx / earlier
    ///     Temps (Div's zero-divisor branch, transcendentals' PLT calls
    ///     and ScalarParam's bounds branch have no pair form);
    ///   - a chain operand may read the pair's OWN store buffer only with
    ///     the store's exact address form (in-place per-cell update — the
    ///     RMW stays self-contained per cell); reads from a buffer any
    ///     OTHER store of the body writes are rejected (unroll-jam
    ///     read-after-write hazard across the two lanes);
    ///   - guards never appear below a packed loop (a +1 lane could exit
    ///     the guarded domain).
    ///
    /// Anything else keeps the scalar form — the gate is all-or-nothing
    /// per loop, and the differential tests hold either way.
    [[nodiscard]] bool analyzePackedLoop(const KernelNode& loop,
                                         const int varDepth) const {
        if (!multiDim) return false;
        // Mirror emitChildList's pairing discipline exactly: a Compute
        // pairs with the IMMEDIATE next sibling when it is a Store; a
        // bare Store child is the walker's no-op; anything else (nested
        // Loop, Guard, ...) disqualifies the loop.
        bool sawPair = false;
        for (std::size_t ci = 0; ci < loop.children.size(); ++ci) {
            const uint32_t cid = loop.children[ci];
            if (cid >= km.nodes.size()) return false;
            const KernelNode& c = km.nodes[cid];
            if (c.op == KernelOp::Store) continue;  // unconsumed: no-op
            if (c.op != KernelOp::Compute) return false;
            if (ci + 1 >= loop.children.size() ||
                loop.children[ci + 1] >= km.nodes.size() ||
                km.nodes[loop.children[ci + 1]].op != KernelOp::Store) {
                return false;  // bare compute: scalar path records it
            }
            const KernelNode& store = km.nodes[loop.children[ci + 1]];
            if (!store.hasAffineStore()) return false;
            if (store.accum != AccumMode::None &&
                store.accum != AccumMode::Add) {
                return false;
            }
            if (coeffAt(store.outIndexCoeffs, varDepth) != 1) return false;
            // Pair temps: chain k lives in slots (2k, 2k+1).
            if (c.exprs.size() > kAsmMaxTempSlots / 2) return false;
            for (std::size_t k = 0; k < c.exprs.size(); ++k) {
                const KernelExpr& e = c.exprs[k];
                if (e.op != MathOp::Add && e.op != MathOp::Sub &&
                    e.op != MathOp::Mul) {
                    return false;
                }
                const KernelOperand* ops[2] = {&e.a, &e.b};
                const std::size_t nOps = isUnaryOp(e.op) ? 1 : 2;
                for (std::size_t oi = 0; oi < nOps; ++oi) {
                    const KernelOperand& o = *ops[oi];
                    switch (o.kind) {  // Rule 78: exhaustive
                        case KernelOperand::Kind::Const:
                            break;
                        case KernelOperand::Kind::Temp:
                            if (o.index >= static_cast<int64_t>(k)) {
                                return false;
                            }
                            break;
                        case KernelOperand::Kind::ElemIdx: {
                            const int64_t coeff =
                                coeffAt(o.idxCoeffs, varDepth);
                            if (coeff != 0 && coeff != 1) return false;
                            if (o.index < 0 ||
                                static_cast<std::size_t>(o.index) >=
                                        km.buffers.size() ||
                                    !bindable[static_cast<std::size_t>(
                                        o.index)]) {
                                return false;
                            }
                            // Buffer-disjointness gate: an operand may
                            // read the pair's OWN store buffer only with
                            // the store's exact address form (in-place
                            // per-cell update); any other overlap with a
                            // written buffer is a cross-lane hazard.
                            for (std::size_t sj = 0;
                                 sj < loop.children.size(); ++sj) {
                                const uint32_t sid = loop.children[sj];
                                if (sid >= km.nodes.size()) return false;
                                const KernelNode& s = km.nodes[sid];
                                if (s.op != KernelOp::Store ||
                                    !s.hasAffineStore()) {
                                    continue;
                                }
                                if (s.bufferOut !=
                                    static_cast<uint32_t>(o.index)) {
                                    continue;
                                }
                                const bool ownStore =
                                    sid == loop.children[ci + 1];
                                const bool sameForm =
                                    ownStore &&
                                    s.outIndexCoeffs == o.idxCoeffs &&
                                    s.outIndexOffset == o.idxOffset;
                                if (!sameForm) return false;
                            }
                            break;
                        }
                        case KernelOperand::Kind::ElemA:
                        case KernelOperand::Kind::ElemB:
                        case KernelOperand::Kind::ScalarParam:
                            return false;
                    }
                }
            }
            sawPair = true;
            ++ci;  // consume the paired store
        }
        return sawPair;
    }

    void scanPackable(uint32_t nodeId, int loopDepth) {
        if (nodeId >= km.nodes.size()) return;
        const KernelNode& n = km.nodes[nodeId];
        const int childDepth =
            n.op == KernelOp::Loop ? loopDepth + 1 : loopDepth;
        if (n.op == KernelOp::Loop) {
            packable[nodeId] = analyzePackedLoop(n, loopDepth);
        }
        for (const uint32_t c : n.children) scanPackable(c, childDepth);
    }

    /// Emits the "poly7" (Sin) family helpers once per module: local
    /// routines mirroring mlk/support/math_families.h OPERATION FOR
    /// OPERATION (Cody-Waite quadrant reduction, degree-13 minimax
    /// Horner evaluation, per-quadrant sin/cos residual selection with
    /// exact sign flips). Constants come from the SAME header (Rule 77:
    /// one coefficient source); floor/fmod go through the PLT exactly
    /// like every other libm reference. Deterministic text (Rule 53).
    void emitPoly7Helpers() {
        // Literal pool entries (RIP-relative; bit-exact doubles).
        const std::string lPiHalf = rodataDouble(families::kPiHalf);
        const std::string lPio2Hi = rodataDouble(families::kPio2Hi);
        const std::string lPio2Lo = rodataDouble(families::kPio2Lo);
        const std::string lHalf = rodataDouble(0.5);
        const std::string lFour = rodataDouble(4.0);
        const std::string lOne = rodataDouble(1.0);
        // Sign flips use the REGISTER-mediated Neg pattern (movabsq +
        // movq + xorpd) — xorpd with a MEMORY operand is a 16-byte SSE
        // access and would require literal-pool alignment the quad pool
        // does not guarantee.
        const std::string lSinC[6] = {rodataDouble(families::kSinC0),
                                      rodataDouble(families::kSinC1),
                                      rodataDouble(families::kSinC2),
                                      rodataDouble(families::kSinC3),
                                      rodataDouble(families::kSinC4),
                                      rodataDouble(families::kSinC5)};
        const std::string lCosC[6] = {rodataDouble(families::kCosC0),
                                      rodataDouble(families::kCosC1),
                                      rodataDouble(families::kCosC2),
                                      rodataDouble(families::kCosC3),
                                      rodataDouble(families::kCosC4),
                                      rodataDouble(families::kCosC5)};
        // Quadrant-reduced residual polynomial (shared shape; the C
        // reference is polySinReduced/polyCosReduced):
        //   r2 = x*x; p = C0; p = p*r2 + Ci (x5); return <combine>
        // The combine step differs per family and is emitted by the
        // caller-provided epilogue lines.
        const auto horner = [&](const std::string (&lc)[6]) {
            text += "    movaps %xmm0, %xmm1\n";
            text += "    mulsd  %xmm1, %xmm1\n";  // r2 = x*x
            text += "    movsd  " + lc[0] + ", %xmm2\n";  // p = C0
            for (int i = 1; i < 6; ++i) {
                text += "    mulsd  %xmm1, %xmm2\n";
                text += "    addsd  " + lc[i] + ", %xmm2\n";
            }
        };
        text += "\n# Family \"poly7\" (Sin): snapshot of "
                "mlk/support/math_families.h — Cody-Waite quadrant\n";
        text += "# reduction + degree-13 minimax residuals; verified "
                "single-digit ULP vs libm on [-pi, pi]\n";
        text += "# (Rule 34 certificate; Rule 77: constants from the "
                "same header).\n";
        text += "    .type   mlk_poly_sin_reduced,@function\n";
        text += "mlk_poly_sin_reduced:\n";
        horner(lSinC);
        // return x + x * r2 * p   (left-associative: x + ((x*r2)*p))
        text += "    movaps %xmm0, %xmm3\n";
        text += "    mulsd  %xmm1, %xmm3\n";
        text += "    mulsd  %xmm2, %xmm3\n";
        text += "    addsd  %xmm3, %xmm0\n";
        text += "    ret\n";
        text += "    .type   mlk_poly_cos_reduced,@function\n";
        text += "mlk_poly_cos_reduced:\n";
        horner(lCosC);
        // return 1.0 - r2 * 0.5 + r2 * r2 * p
        text += "    movsd  " + lHalf + ", %xmm3\n";
        text += "    mulsd  %xmm1, %xmm3\n";   // r2*0.5
        text += "    movsd  " + lOne + ", %xmm4\n";
        text += "    subsd  %xmm3, %xmm4\n";   // 1.0 - r2*0.5
        text += "    movaps %xmm1, %xmm5\n";
        text += "    mulsd  %xmm1, %xmm5\n";   // r2*r2
        text += "    mulsd  %xmm2, %xmm5\n";   // (r2*r2)*p
        text += "    addsd  %xmm5, %xmm4\n";
        text += "    movaps %xmm4, %xmm0\n";
        text += "    ret\n";
        text += "    .type   mlk_poly_sin,@function\n";
        text += "mlk_poly_sin:\n";
        text += "    pushq  %rbp\n";
        text += "    movq   %rsp, %rbp\n";
        text += "    subq   $32, %rsp\n";  // x:-8  n:-16  r:-24 (aligned)
        text += "    movsd  %xmm0, -8(%rbp)\n";  // save x
        // n = floor(x / kPiHalf + 0.5)
        text += "    divsd  " + lPiHalf + ", %xmm0\n";
        text += "    addsd  " + lHalf + ", %xmm0\n";
        text += "    call   floor@PLT\n";
        text += "    movsd  %xmm0, -16(%rbp)\n";
        // r = (x - n*kPio2Hi) - n*kPio2Lo
        text += "    movsd  -8(%rbp), %xmm1\n";
        text += "    movsd  -16(%rbp), %xmm2\n";
        text += "    mulsd  " + lPio2Hi + ", %xmm2\n";
        text += "    subsd  %xmm2, %xmm1\n";
        text += "    movsd  -16(%rbp), %xmm2\n";
        text += "    mulsd  " + lPio2Lo + ", %xmm2\n";
        text += "    subsd  %xmm2, %xmm1\n";
        text += "    movsd  %xmm1, -24(%rbp)\n";
        // q = (int) fmod(n, 4.0); quadrant = q < 0 ? q + 4 : q
        text += "    movsd  -16(%rbp), %xmm0\n";
        text += "    movsd  " + lFour + ", %xmm1\n";
        text += "    call   fmod@PLT\n";
        text += "    cvttsd2si %xmm0, %eax\n";
        text += "    testl  %eax, %eax\n";
        text += "    jns    .Lmlkq_ok\n";
        text += "    addl   $4, %eax\n";
        text += ".Lmlkq_ok:\n";
        // Per-quadrant residual selection (Rule 78: exhaustive).
        text += "    testl  %eax, %eax\n";
        text += "    je     .Lmlkq0\n";
        text += "    cmpl   $1, %eax\n";
        text += "    je     .Lmlkq1\n";
        text += "    cmpl   $2, %eax\n";
        text += "    je     .Lmlkq2\n";
        // q == 3: -polyCosReduced(r)
        text += "    movsd  -24(%rbp), %xmm0\n";
        text += "    call   mlk_poly_cos_reduced\n";
        text += "    movabsq $-9223372036854775808, %rax\n";
        text += "    movq   %rax, %xmm7\n";
        text += "    xorpd  %xmm7, %xmm0\n";
        text += "    jmp    .Lmlkq_done\n";
        text += ".Lmlkq0:\n";
        text += "    movsd  -24(%rbp), %xmm0\n";
        text += "    call   mlk_poly_sin_reduced\n";
        text += "    jmp    .Lmlkq_done\n";
        text += ".Lmlkq1:\n";
        text += "    movsd  -24(%rbp), %xmm0\n";
        text += "    call   mlk_poly_cos_reduced\n";
        text += "    jmp    .Lmlkq_done\n";
        text += ".Lmlkq2:\n";
        text += "    movsd  -24(%rbp), %xmm0\n";
        text += "    call   mlk_poly_sin_reduced\n";
        text += "    movabsq $-9223372036854775808, %rax\n";
        text += "    movq   %rax, %xmm7\n";
        text += "    xorpd  %xmm7, %xmm0\n";
        text += ".Lmlkq_done:\n";
        text += "    leave\n";
        text += "    ret\n";
    }
};

}  // namespace

Result<std::string> emitAsmSource(const KernelModule& kernel,
                                  SymbolTable& symbols) {
    if (kernel.nodes.empty()) {
        return err(ErrorCode::InvalidGraph,
                   "cannot emit empty kernel module");
    }

    AsmEmitter em{kernel, symbols};
    em.multiDim = isMultiDimModule(kernel);

    // Pre-scan: nothing shapes the signature at runtime — the buffer
    // table and the fixed scalars tail fix the arity. Frame pre-scan
    // over the forest roots follows.
    // Argument table: one (ptr[, dims]) pair per bindable buffer in
    // table order (multi-dim), or one pointer per buffer (legacy 1-D);
    // the scalars pair is part of the FIXED ABI (always last).
    em.bindable.assign(kernel.buffers.size(), false);
    em.ptrArgIdx.assign(kernel.buffers.size(), -1);
    int nextArg = 0;
    for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
        const KernelBuffer& b = kernel.buffers[bid];
        if (b.isTemp) {
            if (!em.multiDim) {
                // The 1-D executor routes every temp module through the
                // multi-dim walker; a legacy-form temp never reaches
                // this emitter (kept as an honest structural check).
                return err(ErrorCode::UnsupportedCapability,
                           "asm emitter: temp buffers belong to the "
                           "multi-dim form (walker parity)");
            }
            // Multi-dim temp ABI: the caller materializes one
            // zero-initialized table per isTemp buffer (the walker's
            // temp model) and passes (ptr, dims) in table order — the
            // artifact is the compiled form of the SAME contract.
        }
        if (!b.isInput && !b.isOutput && !b.isTemp) continue;
        em.bindable[bid] = true;
        em.ptrArgIdx[bid] = nextArg;
        nextArg += em.multiDim ? 2 : 1;
    }
    if (nextArg == 0) {
        return err(ErrorCode::InvalidGraph,
                   "asm emitter: module has no bindable buffers");
    }
    // Fixed ABI tail: the scalars pair is always last; the legacy form
    // additionally carries the elements count n right before it.
    nextArg += em.multiDim ? 2 : 3;
    em.totalArgs = nextArg;

    // Frame pre-scan over the forest roots.
    for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
        bool referenced = false;
        for (const KernelNode& n : kernel.nodes) {
            for (const uint32_t c : n.children) {
                referenced = referenced || c == i;
            }
        }
        if (referenced) continue;
        auto r = em.scanFrame(i, 0);
        if (!r.has_value()) {
            return std::unexpected<Error>(r.error());
        }
    }
    if (em.maxTemps > static_cast<int>(kAsmMaxTempSlots)) {
        // Emission would reject the offending chain; reject up front so
        // the frame pre-scan never sizes an unbuildable artifact.
        return err(ErrorCode::InvalidGraph,
                   "asm emitter: compute chain exceeds the temp-slot "
                   "cap");
    }

    // Packed-2 pre-pass (bit-exactness gates in analyzePackedLoop). The
    // frame reserves PAIR temp slots (2 per chain temp) and the two
    // packed-loop control slots whenever any loop qualifies; the packed
    // paths stay disabled when that would not fit the slot cap.
    em.packable.assign(kernel.nodes.size(), false);
    for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
        bool referenced = false;
        for (const KernelNode& n : kernel.nodes) {
            for (const uint32_t c : n.children) {
                referenced = referenced || c == i;
            }
        }
        if (referenced) continue;
        em.scanPackable(i, 0);
    }
    bool anyPackable = false;
    for (const bool p : em.packable) anyPackable = anyPackable || p;
    em.packingEnabled =
        anyPackable && 2 * em.maxTemps <= static_cast<int>(kAsmMaxTempSlots);
    em.frameTemps = em.packingEnabled ? 2 * em.maxTemps : em.maxTemps;
    em.vecSlots = em.packingEnabled ? 2 : 0;

    // Frame: var slots + temp slots + one call-surviving scratch double
    // + packed-loop control slots (when packing) + one HOME SLOT per
    // argument (spilled in the prologue — argument registers are
    // caller-saved and do not survive PLT calls), rounded to the
    // 16-byte ABI alignment.
    const int64_t slots = static_cast<int64_t>(em.maxDepth + em.frameTemps +
                                               em.vecSlots + 1 +
                                               em.totalArgs);
    const int64_t frame = ((slots * 8 + 15) / 16) * 16;

    // Function prologue. The header comment block is assembled AFTER the
    // body walk (it records the realized parallel/simd counts — Rule 148
    // reporting reflects the artifact, not an intention).
    em.text += "    .text\n";
    em.text += "    .globl  mlk_kernel\n";
    em.text += "    .type   mlk_kernel,@function\n";
    em.text += "mlk_kernel:\n";
    em.text += "    pushq   %rbp\n";
    em.text += "    movq    %rsp, %rbp\n";
    if (frame > 0) {
        em.text += "    subq    $" + i64(frame) + ", %rsp\n";
    }

    // Argument home spill: register args move to their frame slots
    // before any body code runs; incoming stack args (k >= 6) are
    // copied through %rax. After this, argument reads are call-safe.
    {
        static const char* kArgRegs[] = {"%rdi", "%rsi", "%rdx",
                                         "%rcx", "%r8",  "%r9"};
        for (int k = 0; k < em.totalArgs; ++k) {
            if (k < 6) {
                em.text += std::string("movq ") + kArgRegs[k] + ", " +
                           em.argSlot(k) + "\n";
            } else {
                em.text += "movq " + i64(16 + 8 * (k - 6)) + "(%rbp), "
                           "%rax\n";
                em.text += "movq %rax, " + em.argSlot(k) + "\n";
            }
        }
    }

    // Body: walk the forest roots.
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
                    const std::string topLbl =
                        em.newLabel("Lloop1d");
                    const std::string endl = em.newLabel("Lloop1dend");
                    // for (v = begin; v < end; ++v) — EXCLUSIVE end
                    // (legacy 1-D semantics; the multi-dim walker is
                    // the inclusive form).
                    em.emitLoadI64(top.begin);
                    em.text +=
                        "movq %rax, " + em.varSlot(0) + "\n";
                    em.text += topLbl + ":\n";
                    if (top.end ==
                        constants::kKernelLoopDynamicBound) {
                        // n = the element count argument (last of the
                        // buffer args, before the fixed scalars pair).
                        const int nArg = em.totalArgs - 3;
                        em.emitLoadArg(nArg, "%rax");
                    } else {
                        em.emitLoadI64(top.end);
                    }
                    em.text += "cmpq %rax, " + em.varSlot(0) + "\n";
                    em.text += "jge " + endl + "\n";
                    em.depth = 1;
                    auto kids = em.emitChildList(top.children, true);
                    if (!kids.has_value()) {
                        return std::unexpected<Error>(kids.error());
                    }
                    em.depth = 0;
                    em.text += "incq " + em.varSlot(0) + "\n";
                    em.text += "jmp " + topLbl + "\n" + endl + ":\n";
                    break;
                }
                case KernelOp::Call:
                    return err(ErrorCode::UnsupportedCapability,
                               "asm emitter: Call nodes must be lowered "
                               "before emission (poly.synth)",
                               121);
                case KernelOp::Guard:
                    return err(ErrorCode::UnsupportedCapability,
                               "asm emitter: speculative guards belong "
                               "to ExecutionEngine (Rule 5)");
                case KernelOp::AllocBuffer:
                case KernelOp::CopyBuffer:
                    return err(ErrorCode::UnsupportedCapability,
                               "asm emitter: buffer management nodes "
                               "are outside the lowering/polyhedral "
                               "emission alphabet");
                case KernelOp::Compute:
                case KernelOp::Store:
                case KernelOp::Load:
                case KernelOp::Barrier:
                case KernelOp::Trace:
                    // "Standalone occurrences are ignored by this
                    // executor on purpose" — mirrored as recorded
                    // comments, not silent drops.
                    em.text += "# top-level " +
                               std::string(kernelOpName(top.op)) +
                               " (no loop context: no semantics)\n";
                    break;
                case KernelOp::kCount:
                    return err(ErrorCode::InvalidGraph,
                               "asm emitter: kCount sentinel");
            }
        }
    }

    // Epilogue: multi-dim artifacts return an ABI status int (0 = ok;
    // 1 = a store address form evaluated negative — the artifact-side
    // mirror of the walker's negative-flat rejection).
    if (em.multiDim) {
        em.text += "    xorl    %eax, %eax\n";
        em.text += ".Lmlk_ret:\n";
        em.text += "    leave\n";
        em.text += "    ret\n";
        em.text += ".Lmlk_neg_store:\n";
        em.text += "    movl    $1, %eax\n";
        em.text += "    leave\n";
        em.text += "    ret\n";
    } else {
        em.text += ".Lmlk_ret:\n";
        em.text += "    leave\n";
        em.text += "    ret\n";
    }

    // Family helpers: local .text routines emitted after the kernel
    // body (still in the .text section), before the literal pool.
    if (em.needsPoly7) {
        em.emitPoly7Helpers();
    }

    // Literal pool + non-executable stack note. The header lands first
    // (comments only), then the function, then the literal pool.
    std::string header;
    const std::string kernelName =
        kernel.name != kInvalidSymbolId
            ? symbols.text(kernel.name)
            : std::string("<anonymous>");
    header += "# Generated by MLK+ (mlk_backend_cpu::emitAsmSource).\n";
    header += "# Kernel: " + kernelName +
              "  hash: " + std::to_string(kernel.hash()) + "\n";
    header += "# Target: x86-64, System V AMD64 ABI, AT&T syntax, "
              "position-independent, SSE2 IEEE-754 doubles";
    if (em.packedLoops > 0) {
        header += " + 2-wide packed innermost loops (movupd/mulpd/addpd)";
    }
    header += ".\n";
    if (em.packedLoops > 0) {
        header += "# Packed-2 execution: " + std::to_string(em.packedLoops) +
                  " innermost loop(s) evaluate two iterations per pass; "
                  "each lane performs exactly the scalar iteration's IEEE "
                  "double ops in the original per-cell order, so results "
                  "are bit-exact by construction; odd trip counts take the "
                  "scalar remainder loop (gates: "
                  "asm_emitter.cpp analyzePackedLoop).\n";
    }
    header += "# Semantics mirror the buffer executor "
              "(runtime/src/execution/kernel_buffers.cpp); differential "
              "verification is bit-exact by construction "
              "(docs/polyhedral_spec.md #backend).\n";
    header += "# Rule 148 recorded: parallel loops=" +
              std::to_string(em.parallelLoops) + ", simd loops=" +
              std::to_string(em.simdLoops) +
              "; this artifact executes sequentially (no runtime "
              "parallelism is linked); parallel rows write disjoint "
              "slabs, so the proven schedule stays deterministic.\n";
    header += "# Rule 96: trace points are recorded as comments. "
              "Rule 5: speculative guards are rejected. Rule 121: Call "
              "nodes must be lowered (poly.synth) before emission.\n";
    header += std::string("# Family: ") +
              (em.needsPoly7
                   ? "poly7 (verified sin family, math_families.h "
                     "certificate; local helper routines)"
                   : "libm") +
              ".\n";
    {
        std::size_t temps = 0;
        for (const KernelBuffer& b : kernel.buffers) {
            temps += b.isTemp ? 1u : 0u;
        }
        if (temps > 0) {
            header += "# Temp ABI: " + std::to_string(temps) +
                      " executor-allocated scratch buffer(s) passed as "
                      "(ptr, dims) table entries — zero-initialized by "
                      "the caller exactly like the walker's temp "
                      "materialization.\n";
        }
    }
    header += "# Assembled out-of-process (ADR-0003/0006): no "
              "in-process machine codegen anywhere in this backend.\n";
    std::string out = header + em.text;
    if (!em.rodata.empty()) {
        out += "    .section .rodata\n";
        out += em.rodata;
    }
    out += "    .section .note.GNU-stack,\"\",@progbits\n";
    return out;
}

}  // namespace mlk
