// SCoP extraction implementation (see scop.h).
//
// Walk shape: walkLoop() validates a Loop node and pushes its bounds;
// walkChildren() pairs Compute nodes with their following Store (the MLK
// kernel statement convention) and dispatches nested loops. Statements
// record raw access coefficients at their OWN depth; a finalize pass
// re-stamps domains and access maps at the full scoping rank once the
// maximum depth is known (shallower statements pin deeper dims to 0).
#include "mlk/poly/scop.h"

#include <string>
#include <utility>

namespace mlk::poly {

PolyWorkspace* createPolyWorkspace() { return new PolyWorkspace{}; }

void destroyPolyWorkspace(PolyWorkspace* ws) noexcept { delete ws; }

namespace {

struct WalkState {
    const KernelModule* kernel{nullptr};
    Scop scop{};
    SmallVector<SymbolId, 8> preorderVars{};  // loop var per nest entry
    std::string reject{};  // cold-path diagnostics (Rule 67)
};

/// True when the node kind can appear inside an affine region.
[[nodiscard]] bool isAffineRegionOp(KernelOp op) noexcept {
    switch (op) {  // Rule 78: exhaustive
        case KernelOp::Loop:
        case KernelOp::Compute:
        case KernelOp::Store:
            return true;
        case KernelOp::Load:
        case KernelOp::AllocBuffer:
        case KernelOp::CopyBuffer:
        case KernelOp::Barrier:
        case KernelOp::Trace:
        case KernelOp::Guard:
        case KernelOp::Call:
        case KernelOp::kCount:
            return false;
    }
    return false;
}

/// Builds a flat-index map from coefficients (per dim) + offset.
[[nodiscard]] AffineMap flatMap(const VarSpace& space,
                                const SmallVector<int64_t, 4>& coeffs,
                                int64_t offset) noexcept {
    AffineMap m;
    m.inSpace = space;
    m.nOut = 1;
    AffineExpr e;
    e.space = space;
    e.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
    for (uint32_t d = 0; d < space.nDims && d < coeffs.size(); ++d) {
        e.coeffs[d] = coeffs[d];
    }
    e.constant = offset;
    m.outputs.push_back(std::move(e));
    return m;
}

[[nodiscard]] bool withinBudgets(WalkState& st, uint32_t depth,
                                 std::size_t accessCount) {
    if (depth > constants::kPolyMaxDims) {
        st.reject = "nest depth exceeds kPolyMaxDims";
        return false;
    }
    if (st.scop.statements.size() >= constants::kPolyMaxStatements) {
        st.reject = "statement count exceeds kPolyMaxStatements";
        return false;
    }
    if (accessCount > constants::kPolyMaxAccessesPerStatement) {
        st.reject = "access count exceeds kPolyMaxAccessesPerStatement";
        return false;
    }
    return true;
}

/// Records one statement from a Compute node + its Store node.
[[nodiscard]] bool recordStatement(WalkState& st,
                                   const SmallVector<int64_t, 8>& loopBegin,
                                   const SmallVector<int64_t, 8>& loopEnd,
                                   const KernelNode& compute,
                                   const KernelNode& store) {
    const uint32_t depth = static_cast<uint32_t>(loopBegin.size());
    if (!withinBudgets(st, depth, compute.exprs.size() + 1)) return false;

    Statement s;
    s.id = static_cast<uint32_t>(st.scop.statements.size());
    s.origOrder = s.id;
    s.depth = depth;

    // Kernel payload: the compute chain verbatim (Temp indices stay valid
    // because the chain is replayed in order by the executor/codegen).
    // Element-wise copies keep sizes obviously bounded for the compiler's
    // allocator range analysis (the small_vector.h suppression does not
    // cover -Walloc-size-larger-than from -Wextra).
    for (const KernelExpr& e : compute.exprs) s.exprs.push_back(e);
    s.storeBuffer = store.bufferOut;
    for (const int64_t c : store.outIndexCoeffs) s.storeCoeffs.push_back(c);
    s.storeOffset = store.outIndexOffset;
    s.accumulate = store.accumulate;
    for (const int64_t v : loopBegin) s.ownLower.push_back(v);
    for (const int64_t v : loopEnd) s.ownUpper.push_back(v - 1);

    // Read accesses from ElemIdx operands (+ legacy ElemA/ElemB).
    for (const KernelExpr& e : compute.exprs) {
        for (const KernelOperand* opnd : {&e.a, &e.b}) {
            if (opnd->kind == KernelOperand::Kind::ElemIdx) {
                MemoryAccess acc;
                acc.bufferId = static_cast<uint32_t>(opnd->index);
                acc.isWrite = false;
                acc.flatIndex = flatMap(VarSpace{depth, 0},
                                        opnd->idxCoeffs, opnd->idxOffset);
                s.accesses.push_back(std::move(acc));
            } else if (opnd->kind == KernelOperand::Kind::ElemA ||
                       opnd->kind == KernelOperand::Kind::ElemB) {
                // Legacy element access: the innermost enclosing loop var.
                if (depth == 0) {
                    st.reject = "legacy element access outside a loop";
                    return false;
                }
                uint32_t buf = compute.bufferA;
                if (opnd->kind == KernelOperand::Kind::ElemB) {
                    buf = compute.bufferB;
                }
                if (buf == constants::kInvalidId) {
                    st.reject = "legacy element operand without a buffer";
                    return false;
                }
                SmallVector<int64_t, 4> legacy(depth, 0);
                legacy[depth - 1] = 1;
                MemoryAccess acc;
                acc.bufferId = buf;
                acc.isWrite = false;
                acc.flatIndex = flatMap(VarSpace{depth, 0}, legacy, 0);
                s.accesses.push_back(std::move(acc));
            }
        }
    }
    // Write access from the store target. Legacy Store nodes carry no
    // affine coefficients: their target is out[i] over the innermost
    // enclosing loop var (kernel_abi.md), so synthesize that coefficient.
    if (s.storeBuffer != constants::kInvalidId) {
        SmallVector<int64_t, 4> wcoeffs = store.outIndexCoeffs;
        if (wcoeffs.empty()) {
            if (depth == 0) {
                st.reject = "legacy store outside a loop";
                return false;
            }
            for (uint32_t d = 0; d < depth; ++d) wcoeffs.push_back(0);
            wcoeffs[depth - 1] = 1;
        }
        s.storeCoeffs = wcoeffs;
        MemoryAccess w;
        w.bufferId = s.storeBuffer;
        w.isWrite = true;
        w.flatIndex = flatMap(VarSpace{depth, 0}, wcoeffs,
                              s.storeOffset);
        s.accesses.push_back(std::move(w));
    }
    st.scop.statements.push_back(std::move(s));
    return true;
}

[[nodiscard]] bool walkChildren(WalkState& st,
                                const SmallVector<uint32_t, 4>& children,
                                SmallVector<uint32_t, 8>& loopVars,
                                SmallVector<int64_t, 8>& loopBegin,
                                SmallVector<int64_t, 8>& loopEnd);

/// Validates and walks a Loop node (bounds already pushed by the caller).
[[nodiscard]] bool walkLoop(WalkState& st, const KernelNode& loop,
                            SmallVector<uint32_t, 8>& loopVars,
                            SmallVector<int64_t, 8>& loopBegin,
                            SmallVector<int64_t, 8>& loopEnd) {
    loopVars.push_back(loop.var);
    st.preorderVars.push_back(loop.var);  // persistent pre-order record
    const bool ok = walkChildren(st, loop.children, loopVars, loopBegin,
                                 loopEnd);
    loopVars.pop_back();
    return ok;
}

/// Walks a children list, pairing Compute nodes with the following Store.
[[nodiscard]] bool walkChildren(WalkState& st,
                                const SmallVector<uint32_t, 4>& children,
                                SmallVector<uint32_t, 8>& loopVars,
                                SmallVector<int64_t, 8>& loopBegin,
                                SmallVector<int64_t, 8>& loopEnd) {
    for (std::size_t i = 0; i < children.size(); ++i) {
        const uint32_t cid = children[i];
        if (cid >= st.kernel->nodes.size()) {
            st.reject = "kernel child node out of range";
            return false;
        }
        const KernelNode& c = st.kernel->nodes[cid];
        if (!isAffineRegionOp(c.op)) {
            st.reject = std::string("kernel op ") + kernelOpName(c.op) +
                        " is not affine-region eligible";
            return false;
        }
        if (c.op == KernelOp::Loop) {
            // Eligibility: unit step, constant bounds (specialized form).
            if (c.step != 1) {
                st.reject = "loop step must be 1 for polyhedral scheduling";
                return false;
            }
            if (c.end < 0 || c.end < c.begin ||
                c.begin < 0) {
                st.reject =
                    "loop bounds must be constant after specialization";
                return false;
            }
            if (static_cast<uint64_t>(c.end) >
                static_cast<uint64_t>(kRationalMagnitudeLimit)) {
                st.reject = "loop bounds exceed engine magnitude limit";
                return false;
            }
            if (loopBegin.size() >= constants::kPolyMaxDims) {
                st.reject = "nest depth exceeds kPolyMaxDims";
                return false;
            }
            loopBegin.push_back(c.begin);
            loopEnd.push_back(c.end);
            const bool ok = walkLoop(st, c, loopVars, loopBegin, loopEnd);
            loopBegin.pop_back();
            loopEnd.pop_back();
            if (!ok) return false;
            continue;
        }
        if (c.op == KernelOp::Compute) {
            if (i + 1 >= children.size() ||
                st.kernel->nodes[children[i + 1]].op != KernelOp::Store) {
                st.reject = "Compute without a paired Store";
                return false;
            }
            const KernelNode& storeNode = st.kernel->nodes[children[i + 1]];
            if (storeNode.bufferOut == constants::kInvalidId) {
                st.reject = "Store without an output buffer";
                return false;
            }
            if (!recordStatement(st, loopBegin, loopEnd, c, storeNode)) {
                return false;
            }
            ++i;  // consume the Store
            continue;
        }
        st.reject = "bare Store outside a Compute+Store pair";
        return false;
    }
    return true;
}

}  // namespace

Result<Scop> extractScop(const KernelModule& kernel, SymbolTable& symbols) {
    (void)symbols;
    if (kernel.nodes.empty()) {
        return err(ErrorCode::InvalidGraph, "kernel module has no nodes");
    }
    WalkState st;
    st.kernel = &kernel;

    // One kernel = one SCoP. The node array is a flat arena; the region
    // roots are the nodes never referenced as children (the executor's
    // flat-forest model, see kernel_ir.h). Every root must be an eligible
    // Loop (mixed kernels stay on the baseline path — see scop.h).
    SmallVector<uint32_t, 8> loopVars;
    SmallVector<int64_t, 8> loopBegin;
    SmallVector<int64_t, 8> loopEnd;
    SmallVector<bool, 8> referenced(kernel.nodes.size(), false);
    for (const KernelNode& n : kernel.nodes) {
        for (const uint32_t c : n.children) {
            if (c < referenced.size()) referenced[c] = true;
        }
    }
    for (uint32_t i = 0; i < kernel.nodes.size(); ++i) {
        if (referenced[i]) continue;
        const KernelNode& root = kernel.nodes[i];
        if (!isAffineRegionOp(root.op)) {
            return err(ErrorCode::UnsupportedCapability,
                       std::string("kernel op ") + kernelOpName(root.op) +
                           " makes the module poly-ineligible");
        }
        if (root.op != KernelOp::Loop) {
            return err(ErrorCode::UnsupportedCapability,
                       "root statements must live inside a loop nest");
        }
        if (root.step != 1 || root.end < 0 || root.end < root.begin ||
            root.begin < 0) {
            return err(ErrorCode::UnsupportedCapability,
                       "root loop bounds must be constant, unit-step");
        }
        if (static_cast<uint64_t>(root.end) >
            static_cast<uint64_t>(kRationalMagnitudeLimit)) {
            return err(ErrorCode::UnsupportedCapability,
                       "root loop bounds exceed engine magnitude limit");
        }
        loopBegin.push_back(root.begin);
        loopEnd.push_back(root.end);
        const bool ok = walkLoop(st, root, loopVars, loopBegin, loopEnd);
        loopBegin.pop_back();
        loopEnd.pop_back();
        if (!ok) {
            return err(ErrorCode::UnsupportedCapability,
                       "SCoP extraction: " + st.reject);
        }
    }
    if (st.scop.statements.empty()) {
        return err(ErrorCode::UnsupportedCapability,
                   "no statements found in the affine region");
    }

    // Finalize: full-rank space; domains from own bounds (deeper dims
    // pinned to 0); access maps and store coefficients padded to rank.
    uint32_t maxDepth = 0;
    for (const Statement& s : st.scop.statements) {
        maxDepth = maxDepth < s.depth ? s.depth : maxDepth;
    }
    st.scop.depth = maxDepth;
    st.scop.space = VarSpace{maxDepth, 0};
    // Dim variables: pre-order nest entry names, truncated to the scoping
    // depth (single-root nests — the poly.synth shape — map exactly).
    st.scop.dimVars.clear();
    for (uint32_t d = 0; d < maxDepth && d < st.preorderVars.size(); ++d) {
        st.scop.dimVars.push_back(st.preorderVars[d]);
    }
    for (Statement& s : st.scop.statements) {
        s.space = st.scop.space;
        // Full-rank domain.
        SmallVector<AffineExpr, 8> lower;
        SmallVector<AffineExpr, 8> upper;
        for (uint32_t d = 0; d < maxDepth; ++d) {
            if (d < s.depth) {
                lower.push_back(
                    AffineExpr::fromConstant(st.scop.space, s.ownLower[d]));
                upper.push_back(
                    AffineExpr::fromConstant(st.scop.space, s.ownUpper[d]));
            } else {
                lower.push_back(AffineExpr::fromConstant(st.scop.space, 0));
                upper.push_back(AffineExpr::fromConstant(st.scop.space, 0));
            }
        }
        auto dom = PresburgerSet::affineBox(st.scop.space, lower, upper);
        if (!dom.has_value()) {
            return err(ErrorCode::Internal,
                       "domain rebuild failed: " + dom.error().message);
        }
        s.domain = std::move(*dom);
        // Pad coefficients to full rank.
        SmallVector<int64_t, 4> storePadded;
        for (uint32_t d = 0; d < maxDepth; ++d) {
            storePadded.push_back(d < s.storeCoeffs.size()
                                      ? s.storeCoeffs[d]
                                      : 0);
        }
        s.storeCoeffs = storePadded;
        for (MemoryAccess& acc : s.accesses) {
            const AffineExpr& e = acc.flatIndex.outputs[0];
            SmallVector<int64_t, 4> padded;
            for (uint32_t d = 0; d < maxDepth; ++d) {
                padded.push_back(e.coeffOf(d));
            }
            acc.flatIndex = flatMap(st.scop.space, padded, e.constant);
        }
    }
    return std::move(st.scop);
}

}  // namespace mlk::poly
