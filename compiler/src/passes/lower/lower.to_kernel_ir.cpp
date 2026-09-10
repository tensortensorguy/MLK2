// lower.to_kernel_ir — Math IR -> Kernel IR (spec §8.11/§12: "Do not emit
// machine code directly from the mathematical graph"). Buffers, fused
// elementwise loops, blocked GEMM calls, and declarative schedule params
// are emitted into the KernelModule sink.
#include "../passes_common.h"
#include "mlk/ir/attrs.h"

#include <functional>

namespace mlk::passes {

namespace {
constexpr Tier kLowerTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class ToKernelIrPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        if (ctx.kernelOut == nullptr) {
            return err(ErrorCode::InvalidArgument,
                       "lower.to_kernel_ir requires kernelOut sink in ctx");
        }
        KernelModule& kernel = *ctx.kernelOut;
        kernel = KernelModule{};
        kernel.name = ctx.symbols->intern("mlk_kernel");
        const SymbolId familyAttr = ctx.symbols->intern("family");
        const SymbolId vwAttr = ctx.symbols->intern("vector_width");
        const SymbolId fusedAttr = ctx.symbols->intern("fusion_group");
        const SymbolId tmAttr = ctx.symbols->intern("tile_m");
        const SymbolId tnAttr = ctx.symbols->intern("tile_n");
        const SymbolId tkAttr = ctx.symbols->intern("tile_k");

        // Buffers: one per placeholder/variable tensor input + output.
        auto addBufferFor = [&](const Value& v, bool input,
                                bool output) -> uint32_t {
            KernelBuffer b;
            b.name = v.name != kInvalidSymbolId
                         ? v.name
                         : ctx.symbols->intern("buf");
            b.dtype = v.type.tensor ? v.type.tensor->element : Dtype::F32;
            b.isInput = input;
            b.isOutput = output;
            if (v.type.tensor) {
                const auto num = v.type.tensor->shape.numel();
                b.elements = num.has_value() ? *num
                                             : constants::kKernelLoopDynamicBound;
                for (std::size_t d = 0; d < v.type.tensor->shape.rank(); ++d) {
                    b.dims.push_back(v.type.tensor->shape.dim(d));
                }
            }
            return kernel.addBuffer(b);
        };

        // Reachability from the outputs over live node inputs. Rule 21
        // keeps superseded equivalent forms in the graph (egraph.extract
        // appends and repoints; nothing is destructively deleted), so the
        // graph may contain orphaned placeholders/values. The KERNEL only
        // realizes the live dataflow — buffers are materialized for
        // reachable inputs/outputs only.
        OpenHashMap<ValueId, uint8_t> reachable;
        {
            SmallVector<ValueId, 16> work;
            for (const ValueId out : graph.outputs()) {
                work.push_back(graph.representative(out));
            }
            while (!work.empty()) {
                const ValueId v = work.back();
                work.pop_back();
                bool inserted = false;
                static_cast<void>(
                    reachable.findOrInsert(v, &inserted, 1));
                if (!inserted) continue;
                const Value& val = graph.value(v);
                if (val.kind == ValueKind::NodeResult) {
                    const Node& p = graph.node(val.producer);
                    for (const ValueId in : p.inputs) {
                        work.push_back(graph.representative(in));
                    }
                }
            }
        }

        // Map value id -> buffer id for tensor inputs/outputs.
        OpenHashMap<ValueId, uint32_t> bufferOf;
        for (const auto& v : graph.values()) {
            if (!v.type.tensor) continue;
            if (v.kind == ValueKind::Placeholder ||
                v.kind == ValueKind::Variable) {
                if (reachable.find(v.id) == nullptr) continue;  // orphaned
                (void)bufferOf.findOrInsert(v.id, nullptr,
                                            addBufferFor(v, true, false));
            }
        }
        SmallVector<uint32_t, 4> outputBuffers;
        for (const ValueId out : graph.outputs()) {
            const Value& v = graph.value(graph.representative(out));
            if (v.type.tensor) {
                const uint32_t bid = addBufferFor(v, false, true);
                (void)bufferOf.findOrInsert(graph.representative(out), nullptr,
                                            bid);
                outputBuffers.push_back(bid);
            }
        }

        // Elementwise fused chains per output; matmul/reduce -> Call.
        for (const ValueId out : graph.outputs()) {
            const Value& outV = graph.value(graph.representative(out));
            if (outV.kind != ValueKind::NodeResult) continue;
            const Node& root = graph.node(outV.producer);

            if (root.op == MathOp::MatMul) {
                // Blocked GEMM kernel via approved runtime entrypoint
                // (Rule 121) with declarative tile params (Rule 54).
                KernelNode call;
                call.op = KernelOp::Call;
                call.math = MathOp::MatMul;
                const Value& a = graph.value(root.inputs[0]);
                const Value& b = graph.value(root.inputs[1]);
                const uint32_t* ba = bufferOf.find(a.id);
                const uint32_t* bb = bufferOf.find(b.id);
                if (ba == nullptr || bb == nullptr) {
                    return err(ErrorCode::InvalidGraph,
                               "matmul operand buffers not materialized");
                }
                call.bufferA = *ba;
                call.bufferB = *bb;
                call.bufferOut = outputBuffers[0];
                // Declarative schedule params (Rule 54): tile attrs flow
                // from schedule.tile into the kernel module.
                for (const SymbolId tileAttr : {tmAttr, tnAttr, tkAttr}) {
                    if (const AttrValue* tv = findAttr(root.attrs, tileAttr)) {
                        if (std::holds_alternative<int64_t>(tv->v)) {
                            (void)kernel.scheduleParams.findOrInsert(
                                tileAttr, nullptr, std::get<int64_t>(tv->v));
                        }
                    }
                }
                (void)kernel.addNode(call);
                r.changed = true;
                continue;
            }

            // Elementwise chain: gather the FULL fused subgraph reachable
            // from the root through ALL inputs (not just input 0 — e.g.
            // Mul(x2, sin(x)) has two elementwise producers). Fusion
            // boundary = fusion_group attr, per the schedule.fuse contract.
            SmallVector<const Node*, 8> chain;  // gather order (root-first)
            const int64_t* rootGroup = nullptr;
            {
                const AttrValue* g = findAttr(root.attrs, fusedAttr);
                rootGroup = g && std::holds_alternative<int64_t>(g->v)
                                ? &std::get<int64_t>(g->v)
                                : nullptr;
            }
            {
                // Worklist over open nodes; seen-set via linear scan (the
                // fused region is bounded by kKernelMaxFusionDepth).
                SmallVector<const Node*, 16> open{&root};
                while (!open.empty()) {
                    const Node* cur = open.back();
                    open.pop_back();
                    bool already = false;
                    for (const Node* c : chain) {
                        if (c == cur) {
                            already = true;
                            break;
                        }
                    }
                    if (already) continue;
                    chain.push_back(cur);
                    if (chain.size() > constants::kKernelMaxFusionDepth) {
                        break;  // budget: keep the region bounded (Rule 10)
                    }
                    for (const ValueId inId : cur->inputs) {
                        const Value& in =
                            graph.value(graph.representative(inId));
                        if (in.kind != ValueKind::NodeResult) continue;
                        const Node* producer = &graph.node(in.producer);
                        if (!isElementwiseMath(producer->op)) continue;
                        const AttrValue* g =
                            findAttr(producer->attrs, fusedAttr);
                        const int64_t* pg =
                            g && std::holds_alternative<int64_t>(g->v)
                                ? &std::get<int64_t>(g->v)
                                : nullptr;
                        // Fusion boundary: the group attr is a scheduling
                        // HINT. Unassigned (null) regions are always
                        // fusible with anything; only an EXPLICITLY
                        // different group is a hard boundary.
                        const bool sameGroup =
                            rootGroup != nullptr && pg != nullptr &&
                            *pg == *rootGroup;
                        const bool anyUngrouped =
                            rootGroup == nullptr || pg == nullptr;
                        if (sameGroup || anyUngrouped) {
                            open.push_back(producer);
                        }
                    }
                }
            }

            // Emit in topological order (producers before consumers) so
            // KernelExpr Temp indices always reference EARLIER exprs.
            SmallVector<const Node*, 8> ordered;
            for (const NodeId nid : graph.topoOrder()) {
                for (const Node* c : chain) {
                    if (c == &graph.node(nid)) {
                        ordered.push_back(c);
                        break;
                    }
                }
            }
            // Fallback: if topo filtering missed (should not happen for a
            // verified graph, Rule 47), keep gather order deterministically.
            if (ordered.size() != chain.size()) ordered = chain;

            // Operand classification per input value of the subgraph.
            // Tensor inputs map to ElemA/ElemB slots in first-touch order;
            // scalar inputs map to ScalarParam indices in first-touch
            // order (documented kernel ABI; see docs/kernel_abi.md).
            OpenHashMap<ValueId, uint32_t> tempOf;    // node result -> expr
            OpenHashMap<ValueId, uint32_t> elemSlot;  // tensor input -> 0/1
            OpenHashMap<ValueId, uint32_t> paramSlot; // scalar input -> idx
            uint32_t nextElemSlot = 0;
            uint32_t nextParamSlot = 0;

            KernelNode compute;
            compute.op = KernelOp::Compute;
            compute.math = ordered.empty() ? MathOp::Add
                                           : ordered.back()->op;
            for (const Node* cn : ordered) {
                KernelExpr e;
                e.op = cn->op;
                std::function<Result<void>(const ValueId,
                                           KernelOperand&)>
                    resolve = [&](const ValueId inId,
                                  KernelOperand& opnd) -> Result<void> {
                    const ValueId rep = graph.representative(inId);
                    if (const uint32_t* t = tempOf.find(rep)) {
                        opnd.kind = KernelOperand::Kind::Temp;
                        opnd.index = static_cast<int64_t>(*t);
                        return {};
                    }
                    const Value& v = graph.value(rep);
                    if (v.kind == ValueKind::NodeResult) {
                        // Rule 39 conversions are transparent at the kernel
                        // boundary: ScalarToTensor lowers to the scalar
                        // operand itself (Const / ScalarParam).
                        const Node& producer = graph.node(v.producer);
                        if (producer.op == MathOp::ScalarToTensor &&
                            !producer.inputs.empty()) {
                            return resolve(producer.inputs[0], opnd);
                        }
                        // Producer outside the fused region: cannot be
                        // inlined (Rule 40: no silent semantic merge).
                        return err(
                            ErrorCode::InvalidGraph,
                            std::string("elementwise kernel input depends "
                                        "on a non-fused node result (op ") +
                                opName(producer.op) + ", producer node " +
                                std::to_string(v.producer) + ", consumer op " +
                                opName(cn->op) + ")");
                    }
                    switch (v.kind) {  // Rule 78: exhaustive
                        case ValueKind::Placeholder:
                        case ValueKind::Variable: {
                            if (v.type.tensor) {
                                if (const uint32_t* s = elemSlot.find(rep)) {
                                    opnd.kind = *s == 0
                                                    ? KernelOperand::Kind::
                                                          ElemA
                                                    : KernelOperand::Kind::
                                                          ElemB;
                                    return {};
                                }
                                if (nextElemSlot >= 2) {
                                    return err(
                                        ErrorCode::InvalidGraph,
                                        "elementwise kernel supports at "
                                        "most 2 distinct tensor inputs");
                                }
                                static_cast<void>(elemSlot.findOrInsert(
                                    rep, nullptr, nextElemSlot));
                                opnd.kind = nextElemSlot == 0
                                                ? KernelOperand::Kind::ElemA
                                                : KernelOperand::Kind::ElemB;
                                ++nextElemSlot;
                                return {};
                            }
                            if (const uint32_t* s = paramSlot.find(rep)) {
                                opnd.kind =
                                    KernelOperand::Kind::ScalarParam;
                                opnd.index = static_cast<int64_t>(*s);
                                return {};
                            }
                            static_cast<void>(paramSlot.findOrInsert(
                                rep, nullptr, nextParamSlot));
                            opnd.kind = KernelOperand::Kind::ScalarParam;
                            opnd.index =
                                static_cast<int64_t>(nextParamSlot);
                            ++nextParamSlot;
                            return {};
                        }
                        case ValueKind::Constant: {
                            opnd.kind = KernelOperand::Kind::Const;
                            opnd.constValue =
                                v.constant.isInt
                                    ? static_cast<double>(v.constant.i64)
                                    : v.constant.f64;
                            return {};
                        }
                        case ValueKind::NodeResult:
                        case ValueKind::Symbol:
                            break;
                    }
                    return err(ErrorCode::InvalidGraph,
                               "unsupported operand kind in elementwise "
                               "kernel");
                };

                const SmallVector<ValueId, 4>& ins = cn->inputs;
                if (ins.size() > 0) {
                    MLK_TRYV(resolve(ins[0], e.a));
                }
                if (ins.size() > 1) {
                    MLK_TRYV(resolve(ins[1], e.b));
                }
                static_cast<void>(tempOf.findOrInsert(
                    graph.representative(cn->results[0]), nullptr,
                    static_cast<uint32_t>(compute.exprs.size())));
                compute.exprs.push_back(e);
            }

            // Tensor input buffers ride on the Compute node (ElemA/ElemB).
            uint32_t elemBuf[2] = {constants::kInvalidId,
                                   constants::kInvalidId};
            elemSlot.forEach([&](ValueId vid, uint32_t slot) {
                if (slot < 2) {
                    if (const uint32_t* b = bufferOf.find(vid)) {
                        elemBuf[slot] = *b;
                    }
                }
            });
            compute.bufferA = elemBuf[0];
            compute.bufferB = elemBuf[1];

            // Build: loop over elements { compute chain; store }.
            KernelNode loop;
            loop.op = KernelOp::Loop;
            loop.var = ctx.symbols->intern("i");
            loop.begin = 0;
            if (outV.type.tensor) {
                const auto num = outV.type.tensor->shape.numel();
                loop.end = num.has_value()
                               ? *num
                               : constants::kKernelLoopDynamicBound;
            }
            int64_t vectorWidth = constants::kDefaultSimdWidthF32;
            for (const Node* cn : ordered) {
                if (const AttrValue* vw = findAttr(cn->attrs, vwAttr)) {
                    if (std::holds_alternative<int64_t>(vw->v)) {
                        vectorWidth = std::get<int64_t>(vw->v);
                    }
                }
            }
            (void)kernel.scheduleParams.findOrInsert(vwAttr, nullptr,
                                                     vectorWidth);

            // Family attr (approx) rides on the compute node for the
            // executor (libm vs poly7 — verified per approx.ulp_verify,
            // Rule 50).
            for (const Node* cn : ordered) {
                if (const AttrValue* fam = findAttr(cn->attrs, familyAttr)) {
                    if (std::holds_alternative<SymbolId>(fam->v)) {
                        compute.family = std::get<SymbolId>(fam->v);
                    }
                }
            }

            KernelNode store;
            store.op = KernelOp::Store;
            store.bufferOut = outputBuffers.empty()
                                  ? constants::kInvalidId
                                  : outputBuffers[0];
            SmallVector<uint32_t, 4> body;
            const uint32_t computeId = kernel.addNode(compute);
            const uint32_t storeId = kernel.addNode(store);
            (void)computeId;
            (void)storeId;
            body.push_back(computeId);
            body.push_back(storeId);
            loop.children = body;
            (void)kernel.addNode(loop);
            r.changed = true;
        }
        return r;
    }
};

void register_lower_to_kernel_ir_pass(SymbolTable& symbols) {
    static ToKernelIrPass pass(symbols, "lower.to_kernel_ir",
                               PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering, {"memory.placed"},
                 {"kernel.built"}, {},
                 {kLowerTiers[0], kLowerTiers[1], kLowerTiers[2]});
}

}  // namespace mlk::passes
