// Physical/memory passes (spec §8.8): memory.liveness, memory.buffer_plan,
// memory.place, memory.layout_select. Rule 23: physical state lives here and
// references the math graph — never the reverse. Rule 40: physicalization
// preserves mathematical meaning.
#include "../passes_common.h"
#include "mlk/core/sparse_set.h"

namespace mlk::passes {

class LivenessPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        // Reverse-order mark liveness over node results (Rule 18: SparseSet
        // for dataflow sets).
        SparseSet live(graph.numValues());
        for (const ValueId out : graph.outputs()) live.insert(out);
        const auto order = graph.topoOrder();
        for (std::size_t i = order.size(); i-- > 0;) {
            const Node& n = graph.node(order[i]);
            bool used = false;
            for (const ValueId res : n.results) used = used || live.contains(res);
            if (used) {
                for (const ValueId in : n.inputs) live.insert(in);
            }
        }
        r.changed = false;
        return r;
    }
};

class BufferPlanPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Every output-tensor value gets a buffer decision; intermediate
        // values inside fused regions stay register-resident (fusion =
        // traffic savings, spec §12).
        const SymbolId bufferAttr = ctx.symbols->intern("buffer");
        const SymbolId alloc = ctx.symbols->intern("allocate");
        for (const ValueId vid : graph.outputs()) {
            const Value& v = graph.value(vid);
            if (!v.type.tensor) continue;
            // buffer decision rides on the node producing it
            if (v.kind != ValueKind::NodeResult) continue;
            Node& n = graph.node(v.producer);
            bool has = false;
            for (const auto& a : n.attrs) {
                if (a.name == bufferAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = bufferAttr;
                a.value = AttrValue{alloc};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

class PlacePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Memory space assignment: host DRAM for the CPU backend (Rule 31:
        // hardware params from Target/HardwareInfo, not hardcoded GPU
        // assumptions).
        const SymbolId spaceAttr = ctx.symbols->intern("memory_space");
        const SymbolId dram = ctx.symbols->intern("dram");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const Value& res = graph.value(n.results[0]);
            if (!res.type.tensor) continue;
            bool has = false;
            for (const auto& a : n.attrs) {
                if (a.name == spaceAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = spaceAttr;
                a.value = AttrValue{dram};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

class LayoutSelectPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Concrete layout: contiguous row-major + 64B alignment (constants;
        // Rule 27/31).
        const SymbolId layoutAttr = ctx.symbols->intern("concrete_layout");
        const SymbolId layout = ctx.symbols->intern("contiguous_aligned");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const Value& res = graph.value(n.results[0]);
            if (!res.type.tensor) continue;
            bool has = false;
            for (const auto& a : n.attrs) {
                if (a.name == layoutAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = layoutAttr;
                a.value = AttrValue{layout};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void registerPhysicalPasses(SymbolTable& symbols) {
    static LivenessPass liveness(symbols, "memory.liveness",
                                 PassKind::Analysis);
    static BufferPlanPass bufferPlan(symbols, "memory.buffer_plan",
                                     PassKind::Transform);
    static PlacePass place(symbols, "memory.place", PassKind::Transform);
    static LayoutSelectPass layoutSelect(symbols, "memory.layout_select",
                                         PassKind::Transform);
    const Tier t12[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
    registerPass(symbols, liveness, PassKind::Analysis, {"effect.inferred"},
                 {"memory.liveness"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, bufferPlan, PassKind::Transform, {"memory.liveness"},
                 {"memory.planned"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, place, PassKind::Transform, {"memory.planned"},
                 {"memory.placed"}, {}, {t12[0], t12[1], t12[2]});
    registerPass(symbols, layoutSelect, PassKind::Transform,
                 {"memory.placed"}, {"memory.layout"}, {},
                 {t12[0], t12[1], t12[2]});
}

}  // namespace mlk::passes
