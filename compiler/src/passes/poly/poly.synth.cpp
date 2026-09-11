// poly.synth — synthesize multi-dim affine Kernel IR from the baseline
// kernel produced by lower.to_kernel_ir (spec §8.13;
// docs/polyhedral_spec.md §synthesis). Lowering pass: the baseline lowers
// tensor work to 1-D flattened elementwise loops and blocked GEMM Call
// nodes; this pass upgrades the affine classes to explicit multi-dim loop
// nests with ElemIdx operands so poly.scop_detect can lift them into
// SCoPs (the "MathGraph -> Kernel IR -> SCoP" chained path).
//
// Recognized classes (everything else stays on the baseline kernel):
//   - Call(MathOp::MatMul) over rank-2 buffers:
//       C[M,N] = A[M,K] * B[K,N]  →  init nest (i,j) + accumulate nest
//       (i,j,k) with row-major flat operands,
//   - 1-D fused elementwise loops whose output buffer has rank >= 2:
//       rebuilt as an R-dim nest with broadcast-aware flat operand
//       coefficients (numpy-style trailing-dim alignment; a dim of size 1
//       in the input contributes coefficient 0),
//   - ReduceSum/elementwise accumulate patterns are roadmap items.
//
// Contract (Rule 142):
//   required  : kernel.built
//   produced  : poly.kernel.synth
//   invalidated: poly.scop, poly.deps, poly.schedule, poly.tile,
//                poly.codegen (any prior polyhedral results)
//   kill switch: "poly.synth" (Rule 60)
#include "../passes_common.h"
#include "mlk/poly/workspace.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};

/// Row-major stride of dim `idx` in a dims vector; 1 for the last dim.
[[nodiscard]] int64_t strideOf(const SmallVector<int64_t, 4>& dims,
                               uint32_t idx) {
    int64_t stride = 1;
    for (uint32_t d = idx + 1; d < dims.size(); ++d) {
        stride *= dims[d] > 0 ? dims[d] : 1;
    }
    return stride;
}

/// Builds the GEMM init+accumulate nest for Call(MatMul) nodes.
[[nodiscard]] bool buildGemmNodes(const KernelModule& km,
                                  const KernelNode& call,
                                  SymbolTable& symbols,
                                  SmallVector<KernelNode, 8>& out) {
    if (call.bufferA >= km.buffers.size() ||
        call.bufferB >= km.buffers.size() ||
        call.bufferOut >= km.buffers.size()) {
        return false;
    }
    const auto& da = km.buffers[call.bufferA].dims;
    const auto& db = km.buffers[call.bufferB].dims;
    if (da.size() != 2 || db.size() != 2) return false;
    const int64_t M = da[0];
    const int64_t K = da[1];
    const int64_t N = db[1];
    if (M <= 0 || K <= 0 || N <= 0) return false;

    // S0: C[i][j] = 0 (flat i*N + j).
    KernelNode compute0;
    compute0.op = KernelOp::Compute;
    KernelExpr zero;
    zero.op = MathOp::Add;  // value carried by the Const operand
    zero.a.kind = KernelOperand::Kind::Const;
    zero.a.constValue = 0.0;
    compute0.exprs.push_back(zero);
    KernelNode store0;
    store0.op = KernelOp::Store;
    store0.bufferOut = call.bufferOut;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{N, 1};
    store0.outIndexOffset = 0;

    // S1: t0 = A[i][k]; t1 = B[k][j]; t2 = t0 * t1; C[i][j] += t2.
    KernelNode compute1;
    compute1.op = KernelOp::Compute;
    KernelExpr ea;
    ea.op = MathOp::Add;
    ea.a.kind = KernelOperand::Kind::ElemIdx;
    ea.a.index = static_cast<int64_t>(call.bufferA);
    ea.a.idxCoeffs = SmallVector<int64_t, 4>{K, 0, 1};  // i*K + k
    KernelExpr eb;
    eb.op = MathOp::Add;
    eb.a.kind = KernelOperand::Kind::ElemIdx;
    eb.a.index = static_cast<int64_t>(call.bufferB);
    eb.a.idxCoeffs = SmallVector<int64_t, 4>{0, 1, N};  // k*N + j
    KernelExpr mul;
    mul.op = MathOp::Mul;
    mul.a.kind = KernelOperand::Kind::Temp;
    mul.a.index = 0;
    mul.b.kind = KernelOperand::Kind::Temp;
    mul.b.index = 1;
    compute1.exprs.push_back(ea);
    compute1.exprs.push_back(eb);
    compute1.exprs.push_back(mul);
    KernelNode store1;
    store1.op = KernelOp::Store;
    store1.bufferOut = call.bufferOut;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{N, 1, 0};
    store1.accumulate = true;

    KernelNode kLoop;
    kLoop.op = KernelOp::Loop;
    kLoop.var = symbols.intern("pk");
    kLoop.begin = 0;
    kLoop.end = K;
    kLoop.children.push_back(out.size() + 0);  // compute1
    kLoop.children.push_back(out.size() + 1);  // store1
    KernelNode jLoop;
    jLoop.op = KernelOp::Loop;
    jLoop.var = symbols.intern("pj");
    jLoop.begin = 0;
    jLoop.end = N;
    jLoop.children.push_back(out.size() + 2);  // compute0
    jLoop.children.push_back(out.size() + 3);  // store0
    jLoop.children.push_back(out.size() + 4);  // kLoop
    KernelNode iLoop;
    iLoop.op = KernelOp::Loop;
    iLoop.var = symbols.intern("pi");
    iLoop.begin = 0;
    iLoop.end = M;
    iLoop.children.push_back(out.size() + 5);  // jLoop

    out.push_back(compute1);
    out.push_back(store1);
    out.push_back(compute0);
    out.push_back(store0);
    out.push_back(kLoop);
    out.push_back(jLoop);
    out.push_back(iLoop);
    return true;
}

/// Rebuilds a 1-D fused elementwise loop as an R-dim nest with
/// broadcast-aware ElemIdx operands. `compute`/`store` are the baseline
/// pair; `loopEnd` is the flattened element count.
[[nodiscard]] bool buildElementwiseNodes(const KernelModule& km,
                                         const KernelNode& /*loop*/,
                                         const KernelNode& compute,
                                         const KernelNode& store,
                                         SymbolTable& symbols,
                                         SmallVector<KernelNode, 8>& out) {
    if (store.bufferOut >= km.buffers.size()) return false;
    const auto& od = km.buffers[store.bufferOut].dims;
    const uint32_t rank = static_cast<uint32_t>(od.size());
    if (rank < 2) return false;  // 1-D baseline is already fine
    for (const int64_t d : od) {
        if (d <= 0) return false;
    }

    // Legacy single-op computes (empty exprs): synthesize the chain.
    SmallVector<KernelExpr, 8> exprs;
    if (compute.exprs.empty()) {
        KernelExpr e;
        e.op = compute.math;
        e.a.kind = KernelOperand::Kind::ElemA;
        e.b.kind = KernelOperand::Kind::ElemB;
        exprs.push_back(e);
    } else {
        for (const KernelExpr& e : compute.exprs) exprs.push_back(e);
    }

    // Map legacy element operands to broadcast-aware ElemIdx.
    auto mapOperand = [&](const KernelOperand& o) -> KernelOperand {
        KernelOperand mapped = o;
        if (o.kind != KernelOperand::Kind::ElemA &&
            o.kind != KernelOperand::Kind::ElemB) {
            return mapped;
        }
        const uint32_t buf = o.kind == KernelOperand::Kind::ElemA
                                 ? compute.bufferA
                                 : compute.bufferB;
        if (buf >= km.buffers.size()) return o;  // unbound: leave legacy
        const auto& id = km.buffers[buf].dims;
        const uint32_t irank = static_cast<uint32_t>(id.size());
        SmallVector<int64_t, 4> coeffs(rank, 0);
        int64_t offset = 0;
        for (uint32_t d = 0; d < rank; ++d) {
            // Trailing-dim alignment.
            if (d + irank < rank) {
                coeffs[d] = 0;  // leading broadcast
                continue;
            }
            const uint32_t aligned = d - (rank - irank);
            if (aligned < id.size() && id[aligned] == 1) {
                coeffs[d] = 0;  // size-1 broadcast
                continue;
            }
            coeffs[d] = strideOf(id, aligned);
        }
        (void)offset;
        mapped.kind = KernelOperand::Kind::ElemIdx;
        mapped.index = static_cast<int64_t>(buf);
        mapped.idxCoeffs = std::move(coeffs);
        mapped.idxOffset = offset;
        return mapped;
    };

    KernelNode newCompute;
    newCompute.op = KernelOp::Compute;
    for (const KernelExpr& e : exprs) {
        KernelExpr mapped;
        mapped.op = e.op;
        mapped.a = mapOperand(e.a);
        mapped.b = mapOperand(e.b);
        newCompute.exprs.push_back(std::move(mapped));
    }
    if (newCompute.exprs.empty()) return false;
    newCompute.family = compute.family;

    // Output flat coefficients.
    SmallVector<int64_t, 4> outCoeffs(rank, 0);
    for (uint32_t d = 0; d < rank; ++d) {
        outCoeffs[d] = strideOf(od, d);
    }
    KernelNode newStore;
    newStore.op = KernelOp::Store;
    newStore.bufferOut = store.bufferOut;
    newStore.outIndexCoeffs = outCoeffs;
    newStore.outIndexOffset = 0;
    newStore.accumulate = store.accumulate;

    // Loop nest over the output dims (outermost first).
    SmallVector<uint32_t, 8> loopIds;
    for (uint32_t d = 0; d < rank; ++d) {
        KernelNode l;
        l.op = KernelOp::Loop;
        l.var = symbols.intern("e" + std::to_string(d));
        l.begin = 0;
        l.end = od[d];
        loopIds.push_back(0);  // ids assigned after all nodes are known
    }
    // Node emission order: compute, store, then loops innermost first so
    // the wiring below can chain outer.children = inner.
    const uint32_t computeId = static_cast<uint32_t>(out.size());
    const uint32_t storeId = computeId + 1;
    out.push_back(newCompute);
    out.push_back(newStore);
    SmallVector<uint32_t, 8> ids;
    ids.push_back(computeId);
    ids.push_back(storeId);
    for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
        KernelNode l;
        l.op = KernelOp::Loop;
        l.var = symbols.intern("e" + std::to_string(d));
        l.begin = 0;
        l.end = od[d];
        const uint32_t lid = static_cast<uint32_t>(out.size());
        ids.push_back(lid);
        out.push_back(std::move(l));
    }
    // ids[2] = innermost loop; ids.back() = outermost loop (the root).
    out[ids[2]].children.push_back(ids[0]);
    out[ids[2]].children.push_back(ids[1]);
    for (uint32_t k = 2; k + 1 < ids.size(); ++k) {
        out[ids[k + 1]].children.push_back(ids[k]);
    }
    return true;
}

}  // namespace

class PolySynthPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        if (ctx.killed(nameId(*ctx.symbols))) return r;  // Rule 60
        if (ctx.kernelOut == nullptr) return r;
        KernelModule& km = *ctx.kernelOut;
        if (km.nodes.empty()) return r;

        // Forest roots (never referenced as children).
        SmallVector<bool, 8> referenced(km.nodes.size(), false);
        for (const KernelNode& n : km.nodes) {
            for (const uint32_t c : n.children) {
                if (c < referenced.size()) referenced[c] = true;
            }
        }
        SmallVector<uint32_t, 4> roots;
        for (uint32_t i = 0; i < km.nodes.size(); ++i) {
            if (!referenced[i]) roots.push_back(i);
        }
        if (roots.size() != 1) return r;  // multi-output: baseline

        SmallVector<KernelNode, 8> rebuilt;
        const KernelNode& root = km.nodes[roots[0]];
        bool ok = false;
        if (root.op == KernelOp::Call && root.math == MathOp::MatMul) {
            ok = buildGemmNodes(km, root, *ctx.symbols, rebuilt);
        } else if (root.op == KernelOp::Loop) {
            // Locate the first Compute+Store pair in the children.
            const KernelNode* compute = nullptr;
            const KernelNode* store = nullptr;
            for (const uint32_t c : root.children) {
                if (c >= km.nodes.size()) return r;
                if (km.nodes[c].op == KernelOp::Compute) {
                    compute = &km.nodes[c];
                } else if (km.nodes[c].op == KernelOp::Store &&
                           compute != nullptr) {
                    store = &km.nodes[c];
                    break;
                }
            }
            if (compute != nullptr && store != nullptr) {
                ok = buildElementwiseNodes(km, root, *compute, *store,
                                           *ctx.symbols, rebuilt);
            }
        }
        if (!ok) return r;  // baseline stays (Rules 62/102)

        km.nodes = std::move(rebuilt);
        if (ctx.polyWorkspace != nullptr) {
            // Invalidate any polyhedral results computed for the old
            // kernel (Rule 30: explicit invalidation).
            ctx.polyWorkspace->scopValid = false;
            ctx.polyWorkspace->dependencesValid = false;
            ctx.polyWorkspace->scheduleValid = false;
            ctx.polyWorkspace->tileValid = false;
            ctx.polyWorkspace->codegenValid = false;
        }
        r.changed = true;
        return r;
    }
};

void register_poly_synth_pass(SymbolTable& symbols) {
    static PolySynthPass pass(symbols, "poly.synth", PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering, {"kernel.built"},
                 {"poly.kernel.synth"},
                 {"poly.scop", "poly.deps", "poly.schedule", "poly.tile",
                  "poly.codegen"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
