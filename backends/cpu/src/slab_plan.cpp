// Slab planning implementation (see slab_plan.h for the soundness
// contract). The walk mirrors the emitters' node discipline exactly:
// loops push per-position var hulls (interval arithmetic over the
// ENCLOSING hulls — never over var names, so every hull bound text
// expands to constants and dims reads only), guards are transparent
// (a superset hull is sound), and Compute chains contribute ElemIdx
// read forms. Structural surprises that the emitters reject anyway
// (non-unit steps, unbound legacy bounds) surface as errors here too,
// so a plan is only ever produced for modules the emitters accept.
#include "mlk/backend/slab_plan.h"

#include <algorithm>
#include <string>
#include <vector>

namespace mlk {

namespace {

[[nodiscard]] std::string i64(const int64_t v) {
    return std::to_string(v);
}

/// A symbolic value: either a folded constant or a host-computable
/// text (constants and dims reads only, by construction). Folding is
/// best-effort: an overflow during constant evaluation degrades to
/// the text form (honest, still exact at run time).
struct Val {
    std::string text{};
    bool isConst{false};
    int64_t v{0};

    [[nodiscard]] static Val constant(const int64_t x) {
        return Val{i64(x), true, x};
    }
    [[nodiscard]] static Val symbolic(std::string t) {
        return Val{std::move(t), false, 0};
    }
    [[nodiscard]] std::string expr() const {
        return isConst ? i64(v) : "(" + text + ")";
    }
};

[[nodiscard]] Val valAdd(const Val& a, const Val& b) {
    if (a.isConst && b.isConst) {
        int64_t r = 0;
        if (!__builtin_add_overflow(a.v, b.v, &r)) {
            return Val::constant(r);
        }
    }
    return Val::symbolic(a.expr() + " + " + b.expr());
}
[[nodiscard]] Val valSub(const Val& a, const Val& b) {
    if (a.isConst && b.isConst) {
        int64_t r = 0;
        if (!__builtin_sub_overflow(a.v, b.v, &r)) {
            return Val::constant(r);
        }
    }
    return Val::symbolic(a.expr() + " - " + b.expr());
}
[[nodiscard]] Val valMulConst(const int64_t c, const Val& a) {
    if (c == 1) return a;
    if (a.isConst) {
        int64_t r = 0;
        if (!__builtin_mul_overflow(c, a.v, &r)) {
            return Val::constant(r);
        }
    }
    return Val::symbolic(i64(c) + "*" + a.expr());
}

/// Symbolic interval [lo, hi].
struct Interval {
    Val lo{};
    Val hi{};
};

/// Min/max folds: constant when both sides fold, runtime select
/// expressions otherwise. Monotone under crossed (statically empty)
/// hulls, so a degenerate form can never shrink the folded hull below
/// the live reads (see the header's soundness note 3).
[[nodiscard]] Val foldMin(const Val& a, const Val& b) {
    if (a.isConst && b.isConst) {
        return Val::constant(a.v < b.v ? a.v : b.v);
    }
    const std::string at = a.expr();
    const std::string bt = b.expr();
    return Val::symbolic("((" + at + ") < (" + bt + ") ? (" + at +
                         ") : (" + bt + "))");
}
[[nodiscard]] Val foldMax(const Val& a, const Val& b) {
    if (a.isConst && b.isConst) {
        return Val::constant(a.v > b.v ? a.v : b.v);
    }
    const std::string at = a.expr();
    const std::string bt = b.expr();
    return Val::symbolic("((" + at + ") > (" + bt + ") ? (" + at +
                         ") : (" + bt + "))");
}

/// Exact interval of an affine form over the enclosing var hulls:
/// each term attains its extrema at the hull corners (coefficients
/// are compile-time constants, so the corner selection is text-time).
[[nodiscard]] Result<Interval> intervalOver(
    const SmallVector<int64_t, 4>& coeffs, const int64_t offset,
    const std::vector<Interval>& hulls) {
    Val lo = Val::constant(offset);
    Val hi = Val::constant(offset);
    for (std::size_t p = 0; p < coeffs.size(); ++p) {
        const int64_t c = coeffs[p];
        if (c == 0) continue;  // zero coefficients never bind a var
        if (p >= hulls.size()) {
            return err(ErrorCode::InvalidGraph,
                       "slab planner: affine form references an unbound "
                       "loop var");
        }
        if (c > 0) {
            lo = valAdd(lo, valMulConst(c, hulls[p].lo));
            hi = valAdd(hi, valMulConst(c, hulls[p].hi));
        } else {
            lo = valSub(lo, valMulConst(-c, hulls[p].hi));
            hi = valSub(hi, valMulConst(-c, hulls[p].lo));
        }
    }
    return Interval{lo, hi};
}

/// One recorded read: the buffer id plus the flat-index form's
/// INTERVAL over the hulls live at the read site (computed during the
/// walk — after it returns, the hull stack is empty again).
struct ReadForm {
    uint32_t buffer{0};
    Interval iv{};
};

struct WalkState {
    std::vector<Interval> hulls{};  // per enclosing var position
    std::vector<ReadForm> reads{};
};

Result<void> walkNode(const KernelModule& km, const uint32_t nodeId,
                      const std::vector<std::string>& dimsName,
                      WalkState& st) {
    if (nodeId >= km.nodes.size()) {
        return err(ErrorCode::InvalidGraph,
                   "slab planner: node id out of range");
    }
    const KernelNode& n = km.nodes[nodeId];
    switch (n.op) {  // Rule 78: exhaustive
        case KernelOp::Loop: {
            // Mirror the emitters' structural rejections so a plan is
            // only produced for modules the emitters accept.
            if (n.step != 1) {
                return err(ErrorCode::UnsupportedCapability,
                           "slab planner: non-unit loop step in a "
                           "multi-dim nest");
            }
            Interval beginIv{};
            if (n.beginCoeffs.empty()) {
                beginIv = Interval{Val::constant(n.begin),
                                   Val::constant(n.begin)};
            } else {
                auto iv = intervalOver(n.beginCoeffs, n.beginOffset,
                                       st.hulls);
                if (!iv.has_value()) {
                    return std::unexpected<Error>(iv.error());
                }
                beginIv = *iv;
            }
            Interval endIv{};
            if (!n.endCoeffs.empty()) {
                auto iv =
                    intervalOver(n.endCoeffs, n.endOffset, st.hulls);
                if (!iv.has_value()) {
                    return std::unexpected<Error>(iv.error());
                }
                endIv = *iv;
            } else if (n.end ==
                           constants::kKernelLoopDynamicBound &&
                       n.endBuf != constants::kInvalidId &&
                       n.endBuf < km.buffers.size() && n.endDim >= 0 &&
                       n.endDim < static_cast<int32_t>(
                                      km.buffers[n.endBuf]
                                          .dims.size())) {
                const std::string e = dimsName[n.endBuf] + "[" +
                                      i64(n.endDim) + "] - 1";
                endIv = Interval{Val::symbolic("(" + e + ")"),
                                 Val::symbolic("(" + e + ")")};
            } else if (n.end == constants::kKernelLoopDynamicBound) {
                return err(ErrorCode::UnsupportedCapability,
                           "slab planner: legacy dynamic bound in a "
                           "multi-dim module (bind endBuf/endDim or "
                           "constant bounds)");
            } else {
                endIv = Interval{Val::constant(n.end - 1),
                                 Val::constant(n.end - 1)};
            }
            // The var's values lie in [begin, inclusive end]; a
            // statically empty loop yields a crossed hull, which the
            // folds handle soundly (superset of the empty set).
            st.hulls.push_back(Interval{beginIv.lo, endIv.hi});
            for (const uint32_t c : n.children) {
                auto r = walkNode(km, c, dimsName, st);
                if (!r.has_value()) {
                    return std::unexpected<Error>(r.error());
                }
            }
            st.hulls.pop_back();
            return {};
        }
        case KernelOp::Guard:
            // Guards narrow execution, never the hulls: a superset
            // hull is sound (the load guard skips hull-excess
            // positions; live reads stay covered).
            for (const uint32_t c : n.children) {
                auto r = walkNode(km, c, dimsName, st);
                if (!r.has_value()) {
                    return std::unexpected<Error>(r.error());
                }
            }
            return {};
        case KernelOp::Compute: {
            for (const KernelExpr& e : n.exprs) {
                const KernelOperand ops[2] = {e.a, e.b};
                for (const KernelOperand& o : ops) {
                    if (o.kind != KernelOperand::Kind::ElemIdx) {
                        continue;
                    }
                    if (o.index < 0 ||
                        static_cast<std::size_t>(o.index) >=
                            km.buffers.size()) {
                        return err(ErrorCode::InvalidGraph,
                                   "slab planner: operand buffer id out "
                                   "of range");
                    }
                    auto iv =
                        intervalOver(o.idxCoeffs, o.idxOffset, st.hulls);
                    if (!iv.has_value()) {
                        return std::unexpected<Error>(iv.error());
                    }
                    st.reads.push_back(
                        ReadForm{static_cast<uint32_t>(o.index), *iv});
                }
            }
            return {};
        }
        case KernelOp::Store:
            // Writes are handled module-wide through `stored`; the
            // store's own index form is not a read.
            return {};
        case KernelOp::Load:
        case KernelOp::Barrier:
        case KernelOp::Trace:
            // No buffer reads, no hull effects (the emitters record
            // these as no-ops too).
            return {};
        case KernelOp::Call:
        case KernelOp::AllocBuffer:
        case KernelOp::CopyBuffer:
            // Emitter-rejected alphabet: mirror the rejection so the
            // planner never plans what the emitter cannot emit.
            return err(ErrorCode::UnsupportedCapability,
                       "slab planner: nodes outside the lowering/"
                       "polyhedral emission alphabet");
        case KernelOp::kCount:
            return err(ErrorCode::InvalidGraph,
                       "slab planner: kCount sentinel");
    }
    return err(ErrorCode::InvalidGraph, "slab planner: unreachable op");
}

}  // namespace

Result<std::vector<RootSlabPlan>> planRootSlabs(
    const KernelModule& kernel, const std::vector<uint32_t>& roots,
    const std::vector<std::string>& dimsName,
    const std::vector<bool>& stored, const SlabEmitOptions& opts) {
    std::vector<RootSlabPlan> plans;
    plans.reserve(roots.size());
    if (!opts.sharedMemSlabs) {
        plans.resize(roots.size());
        return plans;
    }
    for (const uint32_t rootId : roots) {
        WalkState st;
        auto w = walkNode(kernel, rootId, dimsName, st);
        if (!w.has_value()) {
            return std::unexpected<Error>(w.error());
        }
        RootSlabPlan plan;
        // Per-buffer fold (buffer-id order for determinism).
        for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
            bool hasReads = false;
            bool first = true;
            Val mn;
            Val mx;
            for (const ReadForm& f : st.reads) {
                if (f.buffer != bid) continue;
                if (first) {
                    mn = f.iv.lo;
                    mx = f.iv.hi;
                    first = false;
                } else {
                    mn = foldMin(mn, f.iv.lo);
                    mx = foldMax(mx, f.iv.hi);
                }
                hasReads = true;
            }
            if (!hasReads) continue;
            if (bid >= stored.size() || stored[bid]) {
                // The read-only proof failed: RECORDED, never silent
                // (Rule 148) — the buffer keeps its global reads.
                plan.notes.push_back(
                    "buffer " +
                    std::to_string(static_cast<int64_t>(bid)) +
                    ": has reads but is a module-wide store target — "
                    "no slab (read-only proof failed)");
                continue;
            }
            Slab s;
            s.buffer = bid;
            s.minText = mn.expr();
            s.spanText = valAdd(valSub(mx, mn), Val::constant(1)).expr();
            // Buffer element count (dims product) — the load's
            // in-bounds guard.
            std::string elems;
            const KernelBuffer& b = kernel.buffers[bid];
            for (std::size_t d = 0; d < b.dims.size(); ++d) {
                const std::string t =
                    dimsName[bid] + "[" + i64(static_cast<int64_t>(d)) +
                    "]";
                elems = elems.empty()
                            ? "(" + t + ")"
                            : "(" + elems + " * " + t + ")";
            }
            s.elemsText = elems.empty() ? std::string("0") : elems;
            plan.slabs.push_back(std::move(s));
        }
        plans.push_back(std::move(plan));
    }
    return plans;
}

}  // namespace mlk
