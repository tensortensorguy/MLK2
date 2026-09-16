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
//   - Call(MathOp::ReduceSum) over rank-2 input / rank-1 output:
//       Y[M] = sum_k A[M,K]  →  init nest (i) + accumulate nest (i,k)
//       (the reduction dim becomes the carried inner dim; i is parallel),
//   - 1-D fused elementwise loops whose output buffer has rank >= 2:
//       rebuilt as an R-dim nest with broadcast-aware flat operand
//       coefficients (numpy-style trailing-dim alignment; a dim of size 1
//       in the input contributes coefficient 0),
//   - Call(MathOp::Softmax) over rank-2 buffers:
//       Y[M,K] = softmax_rows(A[M,K])  →  the stable-form statement chain
//       with TEMP BUFFER MATERIALIZATION (rowmax/exp/sum temps declared
//       isTemp in the buffer table): S0 RM[m] = -inf; S1 RM[m] max= x
//       (AccumMode::Max); S2 E = exp(x - RM[m]); S3 S[m] = 0;
//       S4 S[m] += E; S5 Y = E / S[m] — four sibling k-bands under m,
//       each an exact op-for-op replay of the reference order
//       (execSoftmaxCall; Rule 90). The scheduler's band-shift roadmap
//       item (per-statement offsets on varying rows) gates FUSED
//       scheduling of this class; until it lands the synthesized nests
//       run as emitted (still exact, still materialized).
//   - elementwise multi-input classes beyond softmax are roadmap items.
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
    store1.accum = AccumMode::Add;

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

/// Builds the ReduceSum init+accumulate nest for Call(ReduceSum) nodes:
/// Y[M] = sum_k A[M,K]. S0: Y[i] = 0 (depth-1 nest); S1: Y[i] += A[i*K+k]
/// (depth-2 nest). The reduction dim is the carried inner dim by the
/// scheduler's dependence analysis (i carries nothing).
[[nodiscard]] bool buildReduceSumNodes(const KernelModule& km,
                                       const KernelNode& call,
                                       SymbolTable& symbols,
                                       SmallVector<KernelNode, 8>& out) {
    if (call.bufferA >= km.buffers.size() ||
        call.bufferOut >= km.buffers.size()) {
        return false;
    }
    const auto& da = km.buffers[call.bufferA].dims;
    const auto& dy = km.buffers[call.bufferOut].dims;
    if (da.size() != 2 || dy.size() != 1) return false;
    const int64_t M = da[0];
    const int64_t K = da[1];
    if (M <= 0 || K <= 0 || dy[0] != M) return false;

    // S1 (accumulate): t0 = A[i][k]; Y[i] += t0. Coefficients are over
    // this statement's own nest dims (i, k).
    KernelNode compute1;
    compute1.op = KernelOp::Compute;
    KernelExpr ea;
    ea.op = MathOp::Add;  // value carried by the ElemIdx operand
    ea.a.kind = KernelOperand::Kind::ElemIdx;
    ea.a.index = static_cast<int64_t>(call.bufferA);
    ea.a.idxCoeffs = SmallVector<int64_t, 4>{K, 1};  // i*K + k
    ea.b.kind = KernelOperand::Kind::Const;
    ea.b.constValue = 0.0;
    compute1.exprs.push_back(ea);
    compute1.bufferA = call.bufferA;
    KernelNode store1;
    store1.op = KernelOp::Store;
    store1.bufferOut = call.bufferOut;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{1, 0};  // Y[i]
    store1.accum = AccumMode::Add;

    // S0 (init): Y[i] = 0 over the depth-1 nest (i).
    KernelNode compute0;
    compute0.op = KernelOp::Compute;
    KernelExpr zero;
    zero.op = MathOp::Add;
    zero.a.kind = KernelOperand::Kind::Const;
    zero.a.constValue = 0.0;
    zero.b.kind = KernelOperand::Kind::Const;
    zero.b.constValue = 0.0;
    compute0.exprs.push_back(zero);
    KernelNode store0;
    store0.op = KernelOp::Store;
    store0.bufferOut = call.bufferOut;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{1};  // Y[i]

    // Nest: i { [S0]; k { [S1] } }.
    KernelNode kLoop;
    kLoop.op = KernelOp::Loop;
    kLoop.var = symbols.intern("rk");
    kLoop.begin = 0;
    kLoop.end = K;
    kLoop.children.push_back(out.size() + 0);  // compute1
    kLoop.children.push_back(out.size() + 1);  // store1
    KernelNode iLoop;
    iLoop.op = KernelOp::Loop;
    iLoop.var = symbols.intern("ri");
    iLoop.begin = 0;
    iLoop.end = M;
    iLoop.children.push_back(out.size() + 3);  // compute0
    iLoop.children.push_back(out.size() + 4);  // store0
    iLoop.children.push_back(out.size() + 2);  // kLoop

    out.push_back(compute1);
    out.push_back(store1);
    out.push_back(kLoop);
    out.push_back(compute0);
    out.push_back(store0);
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
    newStore.accum = store.accum;

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

/// Builds the stable-form softmax statement chain for Call(Softmax)
/// nodes: Y[M,K] = softmax_rows(A[M,K]). Six statements over four
/// sibling k-bands under m, with three materialized temp buffers
/// (rowmax [M], exp [M,K], sum [M]). Every chain replays the reference
/// kernel's operations in the reference order (execSoftmaxCall; the
/// Sub(x, 0) form is the EXACT identity passthrough — unlike Add it
/// preserves -0.0 — so the differential stays bit-exact by
/// construction, Rule 90).
[[nodiscard]] bool buildSoftmaxNodes(KernelModule& km,
                                     const KernelNode& call,
                                     SymbolTable& symbols,
                                     SmallVector<KernelNode, 8>& out) {
    if (call.bufferA >= km.buffers.size() ||
        call.bufferOut >= km.buffers.size()) {
        return false;
    }
    const auto& da = km.buffers[call.bufferA].dims;
    const auto& dy = km.buffers[call.bufferOut].dims;
    if (da.size() != 2 || dy.size() != 2) return false;
    const int64_t M = da[0];
    const int64_t K = da[1];
    if (M <= 0 || K <= 0 || dy[0] != M || dy[1] != K) return false;

    // Temp buffer table: rowmax [M], exp [M,K], sum [M] (Rule 24: the
    // temps are part of the serializable module).
    KernelBuffer rm;
    rm.name = symbols.intern("softmax_rowmax");
    rm.isTemp = true;
    rm.dims = SmallVector<int64_t, 4>{M};
    rm.elements = M;
    KernelBuffer eb;
    eb.name = symbols.intern("softmax_exp");
    eb.isTemp = true;
    eb.dims = SmallVector<int64_t, 4>{M, K};
    eb.elements = M * K;
    KernelBuffer sb;
    sb.name = symbols.intern("softmax_sum");
    sb.isTemp = true;
    sb.dims = SmallVector<int64_t, 4>{M};
    sb.elements = M;
    const uint32_t bufRM = km.addBuffer(rm);
    const uint32_t bufE = km.addBuffer(eb);
    const uint32_t bufS = km.addBuffer(sb);

    auto elem = [&](uint32_t bid, SmallVector<int64_t, 4> coeffs) {
        KernelOperand o;
        o.kind = KernelOperand::Kind::ElemIdx;
        o.index = static_cast<int64_t>(bid);
        o.idxCoeffs = std::move(coeffs);
        return o;
    };
    auto konst = [&](double v) {
        KernelOperand o;
        o.kind = KernelOperand::Kind::Const;
        o.constValue = v;
        return o;
    };
    SmallVector<KernelExpr, 8> es;
    auto pushExpr = [&](MathOp op, KernelOperand a, KernelOperand b) {
        KernelExpr e;
        e.op = op;
        e.a = std::move(a);
        e.b = std::move(b);
        es.push_back(std::move(e));
    };
    auto compute = [&](SymbolId family) {
        KernelNode c;
        c.op = KernelOp::Compute;
        c.family = family;
        return c;
    };
    auto store = [&](uint32_t bid, SmallVector<int64_t, 4> coeffs) {
        KernelNode s;
        s.op = KernelOp::Store;
        s.bufferOut = bid;
        s.outIndexCoeffs = std::move(coeffs);
        return s;
    };

    // S0 (depth 1): RM[m] = -inf.  Add(-inf, 0.0) == -inf exactly.
    KernelNode c0 = compute(call.family);
    pushExpr(MathOp::Add, konst(-INFINITY), konst(0.0));
    c0.exprs = es;
    es.clear();
    KernelNode s0 = store(bufRM, SmallVector<int64_t, 4>{1});  // RM[m]

    // S1 (depth 2): RM[m] max= Sub(x[m,k], 0)  (exact x passthrough).
    KernelNode c1 = compute(call.family);
    pushExpr(MathOp::Sub, elem(call.bufferA, {K, 1}), konst(0.0));
    c1.exprs = es;
    es.clear();
    KernelNode s1 = store(bufRM, SmallVector<int64_t, 4>{1, 0});  // RM[m]
    s1.accum = AccumMode::Max;

    // S2 (depth 2): E[m,k] = Exp(Sub(x[m,k], RM[m])).
    KernelNode c2 = compute(call.family);
    pushExpr(MathOp::Sub, elem(call.bufferA, {K, 1}), elem(bufRM, {1, 0}));
    pushExpr(MathOp::Exp, KernelOperand{KernelOperand::Kind::Temp, 0},
             konst(0.0));
    c2.exprs = es;
    es.clear();
    KernelNode s2 = store(bufE, SmallVector<int64_t, 4>{K, 1});  // E[m,k]

    // S3 (depth 1): S[m] = 0.
    KernelNode c3 = compute(call.family);
    pushExpr(MathOp::Add, konst(0.0), konst(0.0));
    c3.exprs = es;
    es.clear();
    KernelNode s3 = store(bufS, SmallVector<int64_t, 4>{1});  // S[m]

    // S4 (depth 2): S[m] += Sub(E[m,k], 0)  (exact passthrough).
    KernelNode c4 = compute(call.family);
    pushExpr(MathOp::Sub, elem(bufE, {K, 1}), konst(0.0));
    c4.exprs = es;
    es.clear();
    KernelNode s4 = store(bufS, SmallVector<int64_t, 4>{1, 0});  // S[m]
    s4.accum = AccumMode::Add;

    // S5 (depth 2): Y[m,k] = Div(E[m,k], S[m]).
    KernelNode c5 = compute(call.family);
    pushExpr(MathOp::Div, elem(bufE, {K, 1}), elem(bufS, {1, 0}));
    c5.exprs = es;
    es.clear();
    KernelNode s5 = store(call.bufferOut, SmallVector<int64_t, 4>{K, 1});

    // Nest: m { S0; kRM { S1 }; kE { S2 }; S3; kS { S4 }; kD { S5 } }.
    // Sibling k-loops keep every band's instances sequential in the
    // emitted form (the walker executes children in order); the SCoP
    // model sees six statements over dims (m, k) whose FUSED scheduling
    // is the band-shift roadmap item — until it lands the synthesized
    // nests run as emitted (still exact, still materialized).
    // Static node ids (fixed emission order):
    //   0:c1 1:s1 2:kRM 3:c2 4:s2 5:kE 6:c3 7:s3 8:kS 9:c4 10:s4
    //   11:kD 12:c5 13:s5 14:c0 15:s0 16:m
    auto kloop = [&](const char* name, uint32_t c, uint32_t s) {
        KernelNode l;
        l.op = KernelOp::Loop;
        l.var = symbols.intern(name);
        l.begin = 0;
        l.end = K;
        l.children.push_back(c);
        l.children.push_back(s);
        return l;
    };
    out.push_back(c1);                                    // 0
    out.push_back(s1);                                    // 1
    out.push_back(kloop("sk_rm", 0, 1));                  // 2
    out.push_back(c2);                                    // 3
    out.push_back(s2);                                    // 4
    out.push_back(kloop("sk_e", 3, 4));                   // 5
    out.push_back(c3);                                    // 6
    out.push_back(s3);                                    // 7
    out.push_back(kloop("sk_s", 9, 10));                  // 8
    out.push_back(c4);                                    // 9
    out.push_back(s4);                                    // 10
    out.push_back(kloop("sk_d", 12, 13));                 // 11
    out.push_back(c5);                                    // 12
    out.push_back(s5);                                    // 13
    out.push_back(c0);                                    // 14
    out.push_back(s0);                                    // 15
    KernelNode mLoop;
    mLoop.op = KernelOp::Loop;
    mLoop.var = symbols.intern("sm");
    mLoop.begin = 0;
    mLoop.end = M;
    mLoop.children = SmallVector<uint32_t, 4>{14, 15, 2, 5, 6, 7, 8, 11};
    out.push_back(std::move(mLoop));                      // 16
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
        } else if (root.op == KernelOp::Call &&
                   root.math == MathOp::ReduceSum) {
            ok = buildReduceSumNodes(km, root, *ctx.symbols, rebuilt);
        } else if (root.op == KernelOp::Call &&
                   root.math == MathOp::Softmax) {
            ok = buildSoftmaxNodes(km, root, *ctx.symbols, rebuilt);
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
