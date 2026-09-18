// EGraph implementation. Budgeted + deterministic (Rules 10, 53, 143).
#include "egraph.h"

#include "mlk/ir/graph_hash.h"

namespace mlk {

namespace {
/// Cost weights (Rule 27: named, table-driven; Rule 55: cost model data).
constexpr double kCostLeaf = 0.0;
constexpr double kCostCheap = 1.0;   // add/sub/mul
constexpr double kCostDiv = 4.0;
constexpr double kCostLibm = 8.0;    // transcendentals
constexpr double kCostGeneric = 2.0;

double opBaseCost(MathOp op) {
    switch (op) {  // Rule 78: exhaustive
        case MathOp::Add: case MathOp::Sub: case MathOp::Mul:
            return kCostCheap;
        case MathOp::Div: return kCostDiv;
        case MathOp::Pow: return kCostLibm;
        case MathOp::Exp: case MathOp::Log: case MathOp::Sin:
        case MathOp::Cos: case MathOp::Tan: case MathOp::Tanh:
        case MathOp::Sqrt: case MathOp::Rsqrt: case MathOp::Erf:
            return kCostLibm;
        case MathOp::Box: return kCostLeaf;
        default: return kCostGeneric;
    }
}
}  // namespace

EClassId EGraph::find(EClassId c) const {
    while (parent_[c] != c) c = parent_[c];  // acyclic by construction
    return c;
}

EClassId EGraph::makeClass() {
    const auto id = static_cast<EClassId>(parent_.size());
    parent_.push_back(id);
    classMembers_.emplace_back();
    ++liveClasses_;
    return id;
}

Result<EClassId> EGraph::internENode(const ENode& e) {
    if (enodes_.size() >= config_.maxNodes) {
        return err(ErrorCode::BudgetExceeded,
                   "e-graph node budget exhausted (Rule 10)", 10);
    }
    const HashValue h = e.hash();
    if (const EClassId* existing = lookup_.find(h)) {
        const ENode& cand = enodes_[classMembers_[*existing][0]];
        if (cand == e) return *existing;
    }
    const auto enodeId = static_cast<uint32_t>(enodes_.size());
    enodes_.push_back(e);
    EClassId c = makeClass();
    classMembers_[c].push_back(enodeId);
    enodeClass_.push_back(c);
    bool inserted = false;
    EClassId* slot = lookup_.findOrInsert(h, &inserted, c);
    *slot = c;
    return c;
}

Result<EClassId> EGraph::addENode(ENode enode) {
    for (auto& child : enode.children) child = find(child);
    return internENode(enode);
}

Result<EClassId> EGraph::unionClasses(EClassId a, EClassId b, SymbolId rule) {
    a = find(a);
    b = find(b);
    if (a == b) return a;
    if (b < a) {
        const EClassId tmp = a;
        a = b;
        b = tmp;
    }
    parent_[b] = a;
    --liveClasses_;
    for (const uint32_t eid : classMembers_[b]) {
        classMembers_[a].push_back(eid);
        enodeClass_[eid] = a;
        if (rule != kInvalidSymbolId) enodes_[eid].createdBy = rule;
    }
    classMembers_[b].clear();
    return a;
}

Result<EClassId> EGraph::buildFromValue(const MathGraph& graph, ValueId v) {
    const Value& val = graph.value(v);
    if (val.kind == ValueKind::NodeResult) {
        const Node& n = graph.node(val.producer);
        ENode e;
        e.op = n.op;
        for (const auto& a : n.attrs) {
            e.attrHash = hashCombine(e.attrHash, hashU64(a.name));
            e.attrHash = hashCombine(e.attrHash, a.value.hash());
        }
        for (const ValueId in : n.inputs) {            MLK_TRY_VAR(child, buildFromValue(graph, in));
            e.children.push_back(child);
        }
        return internENode(e);
    }
    // Leaf: constant/variable/symbol/placeholder.
    ENode e;
    e.op = MathOp::Box;
    e.attrHash = hashCombine(hashU64(static_cast<uint64_t>(val.kind)),
                             hashU64(val.name));
    e.attrHash = hashCombine(e.attrHash, val.constant.hash());    MLK_TRY_VAR(c, internENode(e));
    // Remember the source value so leaves can be compared/materialized.
    if (leafSource_.find(c) == nullptr) {
        EClassId* slot = leafSource_.findOrInsert(c, nullptr, v);
        *slot = v;
    }
    return c;
}

Result<EClassId> EGraph::importGraph(const MathGraph& graph, ValueId root) {
    imported_ = &graph;
    importedRoot_ = root;    MLK_TRY_VAR(c, buildFromValue(graph, root));
    rootClass_ = c;
    return c;
}

void EGraph::rebuildLookup() {
    lookup_.clear();
    for (std::size_t c = 0; c < parent_.size(); ++c) {
        const EClassId canonical = find(static_cast<EClassId>(c));
        for (const uint32_t eid : classMembers_[canonical]) {
            const HashValue h = enodes_[eid].hash();
            EClassId* slot = lookup_.findOrInsert(h, nullptr, canonical);
            if (*slot != canonical) {
                // Structurally identical enodes in different classes ->
                // merge (congruence closure step).
                (void)unionClasses(*slot, canonical, kInvalidSymbolId);
                rebuildLookup();
                return;
            }
        }
    }
}

Result<bool> EGraph::saturateOnce(SymbolTable& symbols) {
    if (saturated_) return false;
    // Rewrite rules (spec §Pass 4). Rules ADD enodes and union classes;
    // nothing is ever removed (Rule 21). Deterministic enode order.
    bool grew = false;
    const std::size_t n = enodes_.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (enodes_.size() >= config_.maxNodes) {
            saturated_ = true;
            break;
        }
        const ENode e = enodes_[i];  // copy: union may realloc internals
        if (e.children.size() != 2) continue;
        const EClassId lc = find(e.children[0]);
        const EClassId rc = find(e.children[1]);

        // Leaf payloads (only leaf classes carry source values).
        const LeafView* lv = nullptr;
        const LeafView* rv = nullptr;
        LeafView lvs;
        LeafView rvs;
        if (const ValueId* src = leafSource_.find(lc)) {
            lvs = leafView(*imported_, *src);
            lv = &lvs;
        }
        if (const ValueId* src = leafSource_.find(rc)) {
            rvs = leafView(*imported_, *src);
            rv = &rvs;
        }
        auto isConstD = [](const LeafView* l, double expect) {
            return l != nullptr && !l->isInt &&
                   l->kind == ValueKind::Constant && l->f64 == expect;
        };

        // Identity elements (definitely 0/1 leaves only; Rule 22).
        // mul(1,x) -> x and add(0,x) -> x: the PARENT class (the class of
        // enode i itself) unifies with the surviving child's class.
        // Uniting the two CHILDREN instead (a previous bug) declared the
        // constant equivalent to the operand — 0.0 + x extracted as
        // add(0,0) = 0 and x * 1.0 as mul(x,x) = x^2.
        if (e.op == MathOp::Mul && isConstD(lv, 1.0)) {
            (void)unionClasses(enodeClass_[i], rc,
                               symbols.intern("egraph.identity"));
            grew = true;
        } else if (e.op == MathOp::Mul && isConstD(rv, 1.0)) {
            (void)unionClasses(enodeClass_[i], lc,
                               symbols.intern("egraph.identity"));
            grew = true;
        } else if (e.op == MathOp::Add && isConstD(lv, 0.0)) {
            (void)unionClasses(enodeClass_[i], rc,
                               symbols.intern("egraph.identity"));
            grew = true;
        } else if (e.op == MathOp::Add && isConstD(rv, 0.0)) {
            (void)unionClasses(enodeClass_[i], lc,
                               symbols.intern("egraph.identity"));
            grew = true;
        }

        // pow(x, 2) <-> mul(x, x) (Rule 90-gated: exact under IEEE).
        if (e.op == MathOp::Pow && profile_.numeric.ieee754 &&
            isConstD(rv, 2.0)) {
            ENode square;
            square.op = MathOp::Mul;
            square.children.push_back(lc);
            square.children.push_back(lc);            MLK_TRY_VAR(sq, internENode(square));
            (void)unionClasses(enodeClass_[i], sq,
                               symbols.intern("egraph.pow_square"));
            grew = true;
        }
    }
    if (grew) rebuildLookup();
    if (numENodes() >= config_.maxNodes) saturated_ = true;
    return grew;
}

double EGraph::subtreeCost(EClassId c, std::size_t depth) const {
    if (depth > constants::kEgraphMaxClassesPerNode) return 1e9;
    c = find(c);
    double best = 1e18;
    for (const uint32_t eid : classMembers_[c]) {
        const ENode& e = enodes_[eid];
        double cost = opBaseCost(e.op);
        for (const EClassId child : e.children) {
            cost += subtreeCost(child, depth + 1);
        }
        if (cost < best) best = cost;
    }
    return best >= 1e18 ? 0.0 : best;
}

Result<ValueId> EGraph::materialize(SymbolTable& symbols, MathGraph& out,
                                    EClassId c,
                                    OpenHashMap<EClassId, ValueId>& memo,
                                    uint32_t depth) {
    (void)symbols;
    if (depth > constants::kMaxEquivalenceDepth) {
        return err(ErrorCode::BudgetExceeded, "extraction depth exceeded", 10);
    }
    c = find(c);
    if (const ValueId* v = memo.find(c)) return *v;
    const ENode* best = nullptr;
    double bestCost = 1e18;
    for (const uint32_t eid : classMembers_[c]) {
        const ENode& e = enodes_[eid];
        double cost = opBaseCost(e.op);
        for (const EClassId child : e.children) {
            cost += subtreeCost(child);
        }
        if (cost < bestCost) {
            bestCost = cost;
            best = &e;
        }
    }
    if (best == nullptr) {
        return err(ErrorCode::Internal, "empty e-class during extraction");
    }
    if (best->op == MathOp::Box) {
        const ValueId* src = leafSource_.find(c);
        if (src == nullptr) {
            return err(ErrorCode::Internal, "leaf class without source value");
        }
        const Value& orig = imported_->value(*src);
        ValueId resolved = kInvalidValueId;
        switch (orig.kind) {  // Rule 78: exhaustive
            case ValueKind::Constant:
                resolved = orig.constant.isInt
                               ? out.addIntConstant(orig.constant.i64,
                                                    orig.type)
                               : out.addConstant(orig.constant.f64, orig.type);
                break;
            case ValueKind::Variable:
                resolved = out.addVariable(orig.name, orig.type);
                break;
            case ValueKind::Placeholder:
                resolved = out.addPlaceholder(orig.name, orig.type);
                break;
            case ValueKind::Symbol:
                resolved = out.addSymbol(orig.name, orig.type);
                break;
            case ValueKind::NodeResult:
                return err(ErrorCode::Internal,
                           "node result inside leaf class");
        }
        (void)memo.findOrInsert(c, nullptr, resolved);
        return resolved;
    }
    SmallVector<ValueId, 4> ins;
    for (const EClassId child : best->children) {        MLK_TRY_VAR(childV, materialize(symbols, out, child, memo, depth + 1));
        ins.push_back(childV);
    }    MLK_TRY_VAR(v, out.addNode(best->op, ins));
    (void)memo.findOrInsert(c, nullptr, v);
    return v;
}

Result<ValueId> EGraph::extract(SymbolTable& symbols, MathGraph& out,
                                EClassId rootClass) {
    OpenHashMap<EClassId, ValueId> memo;
    return materialize(symbols, out, rootClass, memo, 0);
}

Result<SmallVector<ValueId, 8>> EGraph::extractMany(SymbolTable& symbols,
                                                    MathGraph& out,
                                                    EClassId rootClass,
                                                    std::size_t maxForms) {
    // compile=INF multi-extraction: cheapest form first, then one form per
    // distinct top-level enode (Rule 53: deterministic, cancellable).
    SmallVector<ValueId, 8> forms;
    OpenHashMap<EClassId, ValueId> memo;    MLK_TRY_VAR(primary, materialize(symbols, out, rootClass, memo, 0));
    forms.push_back(primary);
    const EClassId root = find(rootClass);
    const uint32_t primaryEid = classMembers_[root][0];
    for (const uint32_t eid : classMembers_[root]) {
        if (forms.size() >= maxForms) break;
        if (eid == primaryEid) continue;
        const ENode e = enodes_[eid];
        if (e.op == MathOp::Box) continue;
        SmallVector<ValueId, 4> ins;
        bool ok = true;
        OpenHashMap<EClassId, ValueId> localMemo;
        for (const EClassId child : e.children) {
            ValueId childV = kInvalidValueId;
            auto r = materialize(symbols, out, child, localMemo, 0);
            if (!r.has_value()) {
                ok = false;
                break;
            }
            childV = *r;
            ins.push_back(childV);
        }
        if (!ok) continue;
        auto added = out.addNode(e.op, ins);
        if (added.has_value()) {
            bool dup = false;
            for (const ValueId f : forms) {
                if (valueHash(out, f) == valueHash(out, *added)) dup = true;
            }
            if (!dup) forms.push_back(*added);
        }
    }
    return forms;
}

}  // namespace mlk
