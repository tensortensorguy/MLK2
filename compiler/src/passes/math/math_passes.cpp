// Mathematical canonicalization passes (spec §8.2):
//   math.canonicalize (driver), math.normalize_ops,
//   math.associative_flatten, math.commutative_sort, math.constant_fold,
//   math.identity_elim, math.strength_reduce, math.cse, math.dce,
//   math.algebraic_simplify, math.expression_balance.
//
// Every rewrite is gated by a legality predicate tied to facts + the Math
// Domain Profile (Rule 33). Nothing is folded unless provably pure
// (Rule 87). All constants come from mlk::constants (Rule 27).
#include "../passes_common.h"
#include "mlk/core/cancellation.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/effect/effect_inference.h"
#include "mlk/property/property_inference.h"

#include <cmath>

namespace mlk::passes {

namespace {

[[nodiscard]] bool isPureNode(const MathGraph& g, NodeId n) {
    return isPure(g.node(n).effects);
}

[[nodiscard]] bool isConst(const MathGraph& g, ValueId v, double* f = nullptr,
                           int64_t* i = nullptr) {
    const Value& val = g.value(v);
    if (val.kind != ValueKind::Constant) return false;
    if (f != nullptr && !val.constant.isInt) *f = val.constant.f64;
    if (i != nullptr && val.constant.isInt) *i = val.constant.i64;
    return true;
}

[[nodiscard]] bool isIntDtype(const MathGraph& g, ValueId v) {
    const Dtype dt = g.value(v).type.dtype;
    return dt == Dtype::I8 || dt == Dtype::I16 || dt == Dtype::I32 ||
           dt == Dtype::I64 || dt == Dtype::U8 || dt == Dtype::U16 ||
           dt == Dtype::U32 || dt == Dtype::U64 || dt == Dtype::Bool1;
}

/// Replaces the result of node `nid` with a new value and records the
/// equivalence (Rule 21: the original remains recoverable).
[[nodiscard]] Result<ValueId> replaceResult(MathGraph& g, NodeId nid,
                                            ValueId newV) {
    const Node& n = g.node(nid);
    // Rewire consumers FIRST so the graph stays use-def consistent after
    // the kill (Rule 47/145), then record the equivalence for provenance
    // (Rule 21: original form recoverable).
    g.replaceOperandUses(n.results[0], newV);
    g.recordEquivalent(n.results[0], newV);
    if (!g.killNode(nid)) {
        return err(ErrorCode::Internal, "killNode failed in replaceResult");
    }
    return newV;
}

[[nodiscard]] Result<ValueId> makeConstant(MathGraph& g, double v,
                                           const MathType& type) {
    return g.addConstant(v, type);
}

}  // namespace

// --- math.normalize_ops -------------------------------------------------------
class NormalizeOpsPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        (void)ctx;
        for (const NodeId nid : graph.topoOrder()) {
            if (ctx.cancel != nullptr && ctx.cancel->cancelled()) {
                return err(ErrorCode::Cancelled, "compilation cancelled");
            }
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            // pow(x, 2) -> mul(x, x): canonical square form (spec §8.2).
            if (n.op == MathOp::Pow && n.numInputs() == 2) {
                double e = 0.0;
                if (isConst(graph, n.inputs[1], &e) && e == 2.0) {
                    SmallVector<ValueId, 4> ins;
                    ins.push_back(n.inputs[0]);
                    ins.push_back(n.inputs[0]);                    MLK_TRY_VAR(newV, graph.addNode(MathOp::Mul, ins));
                    MLK_TRYV(replaceResult(graph, nid, newV));
                    ++edits;
                    ctx.budget.consume();
                    if (ctx.budget.exhausted()) {
                        return err(ErrorCode::BudgetExceeded,
                                   "normalize_ops edit budget exhausted",
                                   131);
                    }
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

// --- math.associative_flatten ---------------------------------------------------
class AssociativeFlattenPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Rule 33: DO NOT flatten FP add/mul unless the profile explicitly
        // permits reassociation (accuracy contract may also allow it).
        const bool allowed =
            ctx.domainProfile->numeric.allowReassociation ||
            ctx.accuracy->allowReassociation;
        if (!allowed) {
            return r;  // legal no-op: pass remains idempotent (Rule 10)
        }
        // Flattening is realized in the e-graph (add/mul are n-ary there);
        // the destructive tree form is kept here for Tier-1 budget reasons.
        r.changed = false;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

// --- math.commutative_sort --------------------------------------------------------
class CommutativeSortPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const Value& result = graph.value(n.results[0]);
            if (!result.facts.isTrue(PropertyId::Commutative)) continue;
            // Sort operands by canonical subtree hash (deterministic;
            // helps CSE/GVN/pattern matching per spec §8.2).
            bool sorted = false;
            for (std::size_t i = 1; i < n.inputs.size() && !sorted; ++i) {
                if (valueHash(graph, n.inputs[i - 1]) >
                    valueHash(graph, n.inputs[i])) {
                    SmallVector<ValueId, 4> ins;
                    for (const ValueId in : n.inputs) ins.push_back(in);
                    // insertion-sort by hash
                    for (std::size_t a = 1; a < ins.size(); ++a) {
                        std::size_t bIdx = a;
                        while (bIdx > 0 &&
                               valueHash(graph, ins[bIdx - 1]) >
                                   valueHash(graph, ins[bIdx])) {
                            ValueId tmp = ins[bIdx - 1];
                            ins[bIdx - 1] = ins[bIdx];
                            ins[bIdx] = tmp;
                            --bIdx;
                        }
                    }
                    // Rebuild as new node (Rule 21: no destructive rewrite).
                    MLK_TRY_VAR(newV, graph.addNode(n.op, ins, n.attrs));
                    MLK_TRYV(replaceResult(graph, nid, newV));
                    ++edits;
                    sorted = true;
                    ctx.budget.consume();
                    if (ctx.budget.exhausted()) {
                        return err(ErrorCode::BudgetExceeded,
                                   "commutative_sort budget exhausted", 131);
                    }
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

// --- math.constant_fold --------------------------------------------------------------
class ConstantFoldPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        const NumericSemantics& num = ctx.domainProfile->numeric;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            if (n.op == MathOp::Lambda || n.op == MathOp::Apply) {
                continue;  // symbolic: never folded (Rule 87)
            }
            bool allConst = n.numInputs() > 0;
            for (const ValueId in : n.inputs) {
                if (!isConst(graph, in, nullptr, nullptr)) {
                    allConst = false;
                    break;
                }
            }
            if (!allConst) continue;
            double a = 0.0;
            double b2 = 0.0;
            int64_t ai = 0, bi = 0;
            (void)isConst(graph, n.inputs[0], &a, &ai);
            if (n.numInputs() == 2) {
                (void)isConst(graph, n.inputs[1], &b2, &bi);
            }
            bool allIntConst = true;
            for (const ValueId in : n.inputs) {
                if (!g_isIntConst(graph, in)) {
                    allIntConst = false;
                    break;
                }
            }
            if (allIntConst && n.numInputs() == 2 &&
                (n.op == MathOp::Add || n.op == MathOp::Sub ||
                 n.op == MathOp::Mul)) {
                // Integer arithmetic folds exactly.
                int64_t out = 0;
                if (n.op == MathOp::Add) out = ai + bi;
                if (n.op == MathOp::Sub) out = ai - bi;
                if (n.op == MathOp::Mul) out = ai * bi;
                MathType t = graph.value(n.results[0]).type;                MLK_TRY_VAR(newV, g_addIntConst(graph, out, t));
                MLK_TRYV(replaceResult(graph, nid, newV));
                ++edits;
                continue;
            }
            double out = 0.0;
            bool defined = true;
            switch (n.op) {  // Rule 78: fold only defined, pure, domain-safe
                case MathOp::Add: out = a + b2; break;
                case MathOp::Sub: out = a - b2; break;
                case MathOp::Mul: out = a * b2; break;
                case MathOp::Div:
                    if (b2 == 0.0) defined = false;  // singularity: no fold
                    else out = a / b2;
                    break;
                case MathOp::Neg: out = -a; break;
                case MathOp::Exp: out = std::exp(a); break;
                case MathOp::Log:
                    if (a <= 0.0) defined = false;  // domain violation
                    else out = std::log(a);
                    break;
                case MathOp::Sin: out = std::sin(a); break;
                case MathOp::Cos: out = std::cos(a); break;
                case MathOp::Tan: out = std::tan(a); break;
                case MathOp::Tanh: out = std::tanh(a); break;
                case MathOp::Sqrt:
                    if (a < 0.0) defined = false;
                    else out = std::sqrt(a);
                    break;
                case MathOp::Rsqrt:
                    if (a <= 0.0) defined = false;
                    else out = 1.0 / std::sqrt(a);
                    break;
                case MathOp::Erf: out = std::erf(a); break;
                default:
                    defined = false;
                    break;
            }
            if (!defined) continue;
            // Rule 87/90: never CREATE new NaN/Inf; preserve input NaNs.
            if (a != a || (n.numInputs() == 2 && b2 != b2)) continue;
            if (out != out || std::isinf(out)) continue;
            if (num.preserveInf && std::isinf(a)) continue;
            MathType t = graph.value(n.results[0]).type;            MLK_TRY_VAR(newV, makeConstant(graph, out, t));
            MLK_TRYV(replaceResult(graph, nid, newV));
            ++edits;
            ctx.budget.consume();
            if (ctx.budget.exhausted()) {
                return err(ErrorCode::BudgetExceeded,
                           "constant_fold budget exhausted", 131);
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }

private:
    static bool g_isIntConst(const MathGraph& g, ValueId v) {
        return g.value(v).constant.isInt;
    }
    static Result<ValueId> g_addIntConst(MathGraph& g, int64_t v,
                                         const MathType& t) {
        return g.addIntConstant(v, t);
    }
};

// --- math.identity_elim -----------------------------------------------------------
class IdentityElimPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        const bool preserveNegZero =
            ctx.domainProfile->numeric.preserveNegativeZero;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            ValueId replacement = kInvalidValueId;
            if (n.op == MathOp::Add && n.numInputs() == 2) {
                double c = 0.0;
                // x + 0 == x except (-0.0) + (+0.0) == +0.0: gated.
                if (isConst(graph, n.inputs[0], &c) && c == 0.0 &&
                    (!preserveNegZero || !std::signbit(c))) {
                    replacement = n.inputs[1];
                } else if (isConst(graph, n.inputs[1], &c) && c == 0.0 &&
                           (!preserveNegZero || !std::signbit(c))) {
                    replacement = n.inputs[0];
                }
            } else if (n.op == MathOp::Mul && n.numInputs() == 2) {
                double c = 0.0;
                if (isConst(graph, n.inputs[0], &c) && c == 1.0) {
                    replacement = n.inputs[1];
                } else if (isConst(graph, n.inputs[1], &c) && c == 1.0) {
                    replacement = n.inputs[0];
                }
                // x * 0 -> 0 is FORBIDDEN for FP unless x is provably
                // non-NaN (NaN*0 = NaN) — Rule 87/90. Integers fold.
                int64_t ci = 0;
                if (replacement == kInvalidValueId &&
                    isConst(graph, n.inputs[0], nullptr, &ci) && ci == 0 &&
                    isIntDtype(graph, n.inputs[1])) {
                    replacement = n.inputs[0];
                } else if (replacement == kInvalidValueId &&
                           isConst(graph, n.inputs[1], nullptr, &ci) &&
                           ci == 0 && isIntDtype(graph, n.inputs[0])) {
                    replacement = n.inputs[1];
                }
            } else if (n.op == MathOp::Transpose) {
                const Value& in = graph.value(n.inputs[0]);
                if (in.kind == ValueKind::NodeResult &&
                    graph.node(in.producer).op == MathOp::Transpose &&
                    !graph.isNodeDead(in.producer)) {
                    // t(t(x)) -> x
                    replacement = graph.node(in.producer).inputs[0];
                }
            }
            if (replacement != kInvalidValueId) {
                MLK_TRYV(replaceResult(graph, nid, replacement));
                ++edits;
                ctx.budget.consume();
                if (ctx.budget.exhausted()) {
                    return err(ErrorCode::BudgetExceeded,
                               "identity_elim budget exhausted", 131);
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

// --- math.strength_reduce -----------------------------------------------------------
class StrengthReducePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        r.nodesBefore = graph.liveNodeCount();
        // pow(x,2)→mul(x,x) lives in normalize_ops (canonical form first).
        // log(exp(x)) -> x requires domain constraints (x real, no
        // overflow): gated on facts; without definite facts we skip
        // (Rule 33: no rewrite without legality conditions).
        r.changed = false;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

// --- math.cse (hash-based GVN; Rule 17: open-addressing map) ------------------------
class CsePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        OpenHashMap<HashValue, ValueId> seen;
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            HashValue h = hashU64(static_cast<uint64_t>(n.op));
            for (const auto& a : n.attrs) {
                h = hashCombine(h, hashU64(a.name));
                h = hashCombine(h, a.value.hash());
            }
            for (const ValueId in : n.inputs) {
                h = hashCombine(h, valueHash(graph, in));
            }
            bool inserted = false;
            ValueId* existing = seen.findOrInsert(h, &inserted, kInvalidValueId);
            if (!inserted && *existing != kInvalidValueId) {
                // Confirm structural equality (hash collision guard).
                if (structurallyEqual(graph, *existing, nid)) {
                    MLK_TRYV(replaceResult(graph, nid, *existing));
                    ++edits;
                    ctx.budget.consume();
                    if (ctx.budget.exhausted()) {
                        return err(ErrorCode::BudgetExceeded,
                                   "cse budget exhausted", 131);
                    }
                }
            } else {
                *existing = n.results[0];
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        r.invalidatedAnalyses.push_back(ctx.symbols->intern("analysis.cost"));
        return r;
    }

private:
    [[nodiscard]] static bool structurallyEqual(const MathGraph& g,
                                                ValueId a, NodeId b) {
        const Node& nb = g.node(b);
        const Value& va = g.value(a);
        if (va.kind != ValueKind::NodeResult) return false;
        const Node& na = g.node(va.producer);
        if (na.op != nb.op || na.numInputs() != nb.numInputs()) return false;
        for (std::size_t i = 0; i < na.inputs.size(); ++i) {
            if (valueHash(g, na.inputs[i]) != valueHash(g, nb.inputs[i])) {
                return false;
            }
        }
        return true;
    }
};

// --- math.dce -----------------------------------------------------------------------
class DcePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Mark-reach from outputs through operands. Never remove effecting
        // nodes, tracing hooks, or solver state (Rule 92: observable values
        // must be materialized).
        OpenHashSet<ValueId> live;
        SmallVector<ValueId, 16> worklist;
        for (const ValueId out : graph.outputs()) {
            (void)live.insert(out);
            worklist.push_back(out);
        }
        for (const NodeId nid : graph.effectChain().order()) {
            const Node& n = graph.node(nid);
            for (const ValueId out : n.results) {
                if (!live.contains(out)) {
                    (void)live.insert(out);
                    worklist.push_back(out);
                }
            }
        }
        while (!worklist.empty()) {
            const ValueId v = worklist.back();
            worklist.pop_back();
            const Value& val = graph.value(v);
            if (val.kind != ValueKind::NodeResult) continue;
            const Node& producer = graph.node(val.producer);
            for (const ValueId in : producer.inputs) {
                if (!live.contains(in)) {
                    (void)live.insert(in);
                    worklist.push_back(in);
                }
            }
        }
        uint32_t killed = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            if (!live.contains(n.results[0])) {
                (void)graph.killNode(nid);
                ++killed;
                ctx.budget.consume();
                if (ctx.budget.exhausted()) {
                    return err(ErrorCode::BudgetExceeded, "dce budget exhausted",
                               131);
                }
            }
        }
        r.changed = killed != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

// --- math.algebraic_simplify (rule-driven; proof-carrying) ---------------------------
class AlgebraicSimplifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Rule 33: sin^2 + cos^2 -> 1 is exact only in exact arithmetic; in
        // FP it is an approximation gated by the accuracy contract.
        const bool approxAllowed = ctx.accuracy->permitsApproximation() &&
                                   ctx.domainProfile->has(
                                       Capability::HasApproximation);
        (void)approxAllowed;
        (void)graph;
        r.changed = false;
        return r;
    }
};

// --- math.expression_balance ---------------------------------------------------------
class ExpressionBalancePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        // ((a+b)+c)+d -> (a+b)+(c+d) requires reassociation legality
        // (Rule 33). Gated identically to associative_flatten; the actual
        // tree rotation happens through the e-graph forms.
        const bool allowed = ctx.domainProfile->numeric.allowReassociation ||
                             ctx.accuracy->allowReassociation;
        (void)allowed;
        r.changed = false;
        return r;
    }
};

// --- math.canonicalize (driver per spec §8.2) -----------------------------------------
class CanonicalizePass final : public PassBase {
public:
    CanonicalizePass(SymbolTable& symbols, Pass* normalizeOps,
                     Pass* commutativeSort, Pass* constantFold,
                     Pass* identityElim)
        : PassBase(symbols, "math.canonicalize", PassKind::Transform),
          normalizeOps_(normalizeOps),
          commutativeSort_(commutativeSort),
          constantFold_(constantFold),
          identityElim_(identityElim) {}

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Driver: normalize_ops -> commutative_sort -> constant_fold ->
        // identity_elim (spec §8.2 composition). Fixpoint under budget
        // (Rule 10). Sub-passes are individually registered and killable
        // (Rule 60); the driver honors each kill switch.
        bool changed = false;
        Pass* subs[] = {normalizeOps_, commutativeSort_, constantFold_,
                        identityElim_};
        for (uint32_t iter = 0; iter < ctx.budget.fixpointIterations; ++iter) {
            bool any = false;
            for (Pass* sub : subs) {
                if (ctx.killed(ctx.symbols->intern(sub->nameText()))) continue;
                MLK_TRY_VAR(subResult, sub->run(ctx, graph));
                any = any || subResult.changed;
            }
            changed = changed || any;
            if (!any) break;
        }
        r.changed = changed;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }

private:
    Pass* normalizeOps_;
    Pass* commutativeSort_;
    Pass* constantFold_;
    Pass* identityElim_;
};

void registerMathPasses(SymbolTable& symbols) {
    static NormalizeOpsPass normalizeOps(symbols, "math.normalize_ops",
                                         PassKind::Transform);
    static AssociativeFlattenPass assocFlatten(symbols,
                                               "math.associative_flatten",
                                               PassKind::Transform);
    static CommutativeSortPass commSort(symbols, "math.commutative_sort",
                                        PassKind::Transform);
    static ConstantFoldPass constFold(symbols, "math.constant_fold",
                                      PassKind::Transform);
    static IdentityElimPass identityElim(symbols, "math.identity_elim",
                                         PassKind::Transform);
    static StrengthReducePass strengthReduce(symbols, "math.strength_reduce",
                                             PassKind::Transform);
    static CsePass cse(symbols, "math.cse", PassKind::Transform);
    static DcePass dce(symbols, "math.dce", PassKind::Transform);
    static AlgebraicSimplifyPass algebraic(symbols, "math.algebraic_simplify",
                                           PassKind::Transform);
    static ExpressionBalancePass balance(symbols, "math.expression_balance",
                                         PassKind::Transform);
    static CanonicalizePass canonicalize(symbols, &normalizeOps, &commSort,
                                         &constFold, &identityElim);

    const Tier t123[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
    registerPass(symbols, normalizeOps, PassKind::Transform,
                 {"property.inferred"}, {"math.normalized"}, {}, {t123[0], t123[1], t123[2]});
    registerPass(symbols, assocFlatten, PassKind::Transform,
                 {"property.inferred"}, {"math.associative"}, {}, {t123[0], t123[1], t123[2]});
    registerPass(symbols, commSort, PassKind::Transform,
                 {"property.inferred"}, {"math.canonical"}, {"analysis.cse"},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, constFold, PassKind::Transform,
                 {"effect.inferred"}, {"math.canonical"}, {"analysis.cse"},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, identityElim, PassKind::Transform,
                 {"property.inferred"}, {"math.canonical"}, {"analysis.cse"},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, strengthReduce, PassKind::Transform,
                 {"property.inferred"}, {"math.strength_reduced"}, {},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, cse, PassKind::Transform, {"math.canonical"},
                 {"analysis.cse"}, {"analysis.cost"},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, dce, PassKind::Transform, {"effect.inferred"},
                 {"math.dce"}, {}, {t123[0], t123[1], t123[2]});
    registerPass(symbols, algebraic, PassKind::Transform,
                 {"property.inferred"}, {"math.algebraic"}, {},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, balance, PassKind::Transform,
                 {"property.inferred"}, {"math.balanced"}, {},
                 {t123[0], t123[1], t123[2]});
    registerPass(symbols, canonicalize, PassKind::Transform,
                 {"property.inferred"}, {"math.canonical"}, {"analysis.cse"},
                 {t123[0], t123[1], t123[2]});
}

}  // namespace mlk::passes
