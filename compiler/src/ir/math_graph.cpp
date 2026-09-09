// MathGraph implementation. See math_graph.h for the design-law mapping.
#include "mlk/ir/math_graph.h"

#include "mlk/effect/effect_inference.h"

namespace mlk {

ValueId MathGraph::allocateValue(Value v) {
    v.id = static_cast<ValueId>(values_.size());
    values_.push_back(std::move(v));
    users_.emplace_back();
    hashCache_.reset();
    return values_.back().id;
}

ValueId MathGraph::addInputLike(ValueKind kind, SymbolId name, MathType type) {
    Value v;
    v.kind = kind;
    v.name = name;
    v.type = std::move(type);
    return allocateValue(std::move(v));
}

ValueId MathGraph::addPlaceholder(SymbolId name, MathType type) {
    return addInputLike(ValueKind::Placeholder, name, std::move(type));
}

ValueId MathGraph::addVariable(SymbolId name, MathType type) {
    return addInputLike(ValueKind::Variable, name, std::move(type));
}

ValueId MathGraph::addSymbol(SymbolId name, MathType type) {
    return addInputLike(ValueKind::Symbol, name, std::move(type));
}

ValueId MathGraph::addConstant(double v, MathType type) {
    Value val;
    val.kind = ValueKind::Constant;
    val.type = std::move(type);
    val.constant = ConstantPayload{false, v, 0};
    val.facts.setTriState(PropertyId::Pure, TriState::True);
    return allocateValue(std::move(val));
}

ValueId MathGraph::addIntConstant(int64_t v, MathType type) {
    Value val;
    val.kind = ValueKind::Constant;
    val.type = std::move(type);
    val.constant = ConstantPayload{true, 0.0, v};
    val.facts.setTriState(PropertyId::Pure, TriState::True);
    return allocateValue(std::move(val));
}

Result<ValueId> MathGraph::addNode(MathOp op,
                                   const SmallVector<ValueId, 4>& inputs,
                                   AttrList attrs) {
    Node n;
    n.id = static_cast<NodeId>(nodes_.size());
    n.op = op;
    for (const ValueId in : inputs) {
        if (in >= values_.size()) {
            return err(ErrorCode::InvalidGraph,
                       "operand value id out of range", 15);
        }
        n.inputs.push_back(in);
    }
    n.attrs = std::move(attrs);

    // Result value: one per node (MVP IR model; multi-result ops would be
    // extended here without changing the law set).
    Value out;
    out.kind = ValueKind::NodeResult;
    out.producer = n.id;
    out.name = symbols_->intern((std::string(opName(op)) + "_out").c_str());

    n.results.push_back(allocateValue(std::move(out)));
    nodes_.push_back(std::move(n));

    // Use-def maintenance
    Node& stored = nodes_.back();
    for (const ValueId in : stored.inputs) {
        users_[in].push_back(stored.id);
    }

    // Effects (Rule 140)
    attachEffects(stored);

    hashCache_.reset();
    ++version_;
    return Result<ValueId>{stored.results[0]};
}

void MathGraph::attachEffects(Node& n) {
    n.effects = inferNodeEffects(*this, n.id);
    if (!isPure(n.effects)) {
        effects_.append(n.id);
    }
}

void MathGraph::rebuildUsers() {
    users_.assign(values_.size(), {});
    for (const auto& n : nodes_) {
        if (n.flags.test(NodeFlag::Dead)) continue;
        for (const ValueId in : n.inputs) {
            users_[in].push_back(n.id);
        }
    }
}

HashValue MathGraph::hash() const {
    if (hashCache_) return *hashCache_;
    // Bottom-up structural hash in topological order (Rule 24: stable,
    // content-derived, independent of node creation order where the math is
    // identical — commutative canonicalization happens in passes, not here).
    HashValue h = kHashSeed;
    for (const auto& n : nodes_) {
        if (n.flags.test(NodeFlag::Dead)) continue;
        HashValue nh = hashU64(static_cast<uint64_t>(n.op));
        for (const auto& a : n.attrs) {
            nh = hashCombine(nh, hashU64(a.name));
            nh = hashCombine(nh, a.value.hash());
        }
        for (const ValueId in : n.inputs) {
            const Value& v = values_[in];
            HashValue vh = v.hashValid
                               ? v.hash
                               : hashCombine(hashU64(static_cast<uint64_t>(v.kind)),
                                             v.type.hash());
            nh = hashCombine(nh, vh);
        }
        const_cast<Value&>(values_[n.results[0]]).hash = nh;
        const_cast<Value&>(values_[n.results[0]]).hashValid = true;
        h = hashCombine(h, nh);
    }
    hashCache_ = h;
    return h;
}

void MathGraph::replaceOperandUses(ValueId oldV, ValueId newV) {
    if (oldV == newV || oldV >= values_.size() || newV >= values_.size()) {
        return;
    }
    for (auto& n : nodes_) {
        if (n.flags.test(NodeFlag::Dead)) continue;
        for (ValueId& in : n.inputs) {
            if (in == oldV) in = newV;
        }
    }
    rebuildUsers();
    hashCache_.reset();
    ++version_;
}

void MathGraph::recordEquivalent(ValueId oldV, ValueId newV) {
    bool inserted = false;
    ValueId* mapped = representative_.findOrInsert(oldV, &inserted, newV);
    *mapped = newV;
    value(oldV).flags.set(ValueFlag::Canonical, false);
    value(newV).flags.set(ValueFlag::Canonical, true);
    ++version_;
}

void MathGraph::markLoweredAway(ValueId v, SymbolId decision) {
    (void)decision;  // recorded by caller in telemetry/proof trail
    (void)loweredAway_.insert(v);
    value(v).flags.set(ValueFlag::Dead, true);
    ++version_;
}

bool MathGraph::killNode(NodeId n) {
    Node& nodeRef = node(n);
    if (nodeRef.flags.test(NodeFlag::Dead)) return false;
    nodeRef.flags.set(NodeFlag::Dead, true);
    effects_.remove(n);
    hashCache_.reset();
    ++version_;
    return true;
}

SmallVector<NodeId, 16> MathGraph::topoOrder() const {
    // Values are appended in creation order and a value's producer always
    // has a lower id than its consumers (no in-place mutation of operands),
    // so node id order IS a valid topological order. Verified by the graph
    // verifier (acyclicity check) rather than trusted blindly here.
    SmallVector<NodeId, 16> order;
    order.reserve(nodes_.size());
    for (const auto& n : nodes_) {
        if (!n.flags.test(NodeFlag::Dead)) order.push_back(n.id);
    }
    return order;
}

}  // namespace mlk
