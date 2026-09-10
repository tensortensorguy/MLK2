// Shared rewrite helpers for the math.* passes (internal; not a pass).
// Kept here so each per-pass file stays self-contained without duplicating
// the use-def discipline (Rule 47/145) and equivalence bookkeeping (Rule 21).
#pragma once

#include "mlk/core/result.h"
#include "mlk/effect/effect_inference.h"
#include "mlk/ir/math_graph.h"

namespace mlk::passes {

[[nodiscard]] inline bool isPureNode(const MathGraph& g, NodeId n) {
    return isPure(g.node(n).effects);
}

[[nodiscard]] inline bool isConst(const MathGraph& g, ValueId v,
                                  double* f = nullptr,
                                  int64_t* i = nullptr) {
    const Value& val = g.value(v);
    if (val.kind != ValueKind::Constant) return false;
    if (f != nullptr && !val.constant.isInt) *f = val.constant.f64;
    if (i != nullptr && val.constant.isInt) *i = val.constant.i64;
    return true;
}

[[nodiscard]] inline bool isIntDtype(const MathGraph& g, ValueId v) {
    const Dtype dt = g.value(v).type.dtype;
    return dt == Dtype::I8 || dt == Dtype::I16 || dt == Dtype::I32 ||
           dt == Dtype::I64 || dt == Dtype::U8 || dt == Dtype::U16 ||
           dt == Dtype::U32 || dt == Dtype::U64 || dt == Dtype::Bool1;
}

/// Replaces the result of node `nid` with a new value and records the
/// equivalence (Rule 21: the original remains recoverable).
[[nodiscard]] inline Result<ValueId> replaceResult(MathGraph& g, NodeId nid,
                                                   ValueId newV) {
    const Node& n = g.node(nid);
    // Rewrite results are rebuilt without an inferred type; transfer the
    // original's type so downstream inference/verification stays sound
    // (the rewrite is equivalence-preserving, so the type carries over).
    const Value& oldVal = g.value(n.results[0]);
    Value& newVal = g.value(newV);
    if (newVal.type.domain == Domain::Unknown) {
        newVal.type = oldVal.type;
    }
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

[[nodiscard]] inline Result<ValueId> makeConstant(MathGraph& g, double v,
                                                  const MathType& type) {
    return g.addConstant(v, type);
}

}  // namespace mlk::passes
