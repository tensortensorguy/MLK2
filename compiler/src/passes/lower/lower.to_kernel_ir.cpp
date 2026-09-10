// lower.to_kernel_ir — Math IR -> Kernel IR (spec §8.11/§12: "Do not emit
// machine code directly from the mathematical graph"). Buffers, fused
// elementwise loops, blocked GEMM calls, and declarative schedule params
// are emitted into the KernelModule sink.
#include "../passes_common.h"
#include "mlk/ir/attrs.h"

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
            }
            return kernel.addBuffer(b);
        };

        // Map value id -> buffer id for tensor inputs/outputs.
        OpenHashMap<ValueId, uint32_t> bufferOf;
        for (const auto& v : graph.values()) {
            if (!v.type.tensor) continue;
            if (v.kind == ValueKind::Placeholder ||
                v.kind == ValueKind::Variable) {
                (void)bufferOf.findOrInsert(v.id, nullptr,
                                            addBufferFor(v, true, false));
            }
        }
        SmallVector<uint32_t, 4> outputBuffers;
        for (const ValueId out : graph.outputs()) {
            const Value& v = graph.value(out);
            if (v.type.tensor) {
                const uint32_t bid = addBufferFor(v, false, true);
                (void)bufferOf.findOrInsert(out, nullptr, bid);
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

            // Elementwise chain: gather ops from root down through
            // fused elementwise producers (fusion boundary = group attr).
            SmallVector<const Node*, 8> chain;
            const Node* cur = &root;
            const int64_t* rootGroup = nullptr;
            {
                const AttrValue* g = findAttr(cur->attrs, fusedAttr);
                rootGroup = g && std::holds_alternative<int64_t>(g->v)
                                ? &std::get<int64_t>(g->v)
                                : nullptr;
            }
            while (cur != nullptr && isElementwiseMath(cur->op)) {
                chain.push_back(cur);
                if (cur->inputs.empty()) break;
                const Value& in = graph.value(graph.representative(cur->inputs[0]));
                if (in.kind != ValueKind::NodeResult) break;
                const Node* producer = &graph.node(in.producer);
                if (!isElementwiseMath(producer->op)) break;
                const AttrValue* g = findAttr(producer->attrs, fusedAttr);
                const int64_t* pg =
                    g && std::holds_alternative<int64_t>(g->v)
                        ? &std::get<int64_t>(g->v)
                        : nullptr;
                // Fusion boundary: different (or missing) fusion group ends
                // the chain — MVP keeps single-op regions unless grouped.
                if (rootGroup != nullptr && pg != nullptr &&
                    *pg == *rootGroup) {
                    cur = producer;
                } else if (rootGroup == nullptr && pg == nullptr &&
                           chain.size() < constants::kKernelMaxFusionDepth) {
                    cur = producer;  // ungrouped chains fuse transitively
                } else {
                    break;
                }
            }

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
            for (const Node* cn : chain) {
                if (const AttrValue* vw = findAttr(cn->attrs, vwAttr)) {
                    if (std::holds_alternative<int64_t>(vw->v)) {
                        vectorWidth = std::get<int64_t>(vw->v);
                    }
                }
            }
            (void)kernel.scheduleParams.findOrInsert(vwAttr, nullptr,
                                                     vectorWidth);

            KernelNode compute;
            compute.op = KernelOp::Compute;
            compute.math = chain.empty() ? MathOp::Add : chain[0]->op;
            // Family attr (approx) rides on the compute node for the
            // executor (libm vs poly7 — verified per approx.ulp_verify,
            // Rule 50).
            for (const Node* cn : chain) {
                if (const AttrValue* fam = findAttr(cn->attrs, familyAttr)) {
                    if (std::holds_alternative<SymbolId>(fam->v)) {
                        compute.family = std::get<SymbolId>(fam->v);
                    }
                }
            }
            compute.bufferA = outputBuffers.empty()
                                  ? constants::kInvalidId
                                  : outputBuffers[0];
            // Inputs: MVP elementwise kernels read buffer 0 (x) and inline
            // constants; chained intermediates fold into the compute body.
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
