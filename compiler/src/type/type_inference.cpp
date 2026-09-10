// Type and shape inference implementation.
// Law 39 enforcement: no implicit conversions. If operand dtypes/shapes
// disagree in ways the op does not define, inference fails with an
// actionable diagnostic naming the offending node.
#include "mlk/type/type_inference.h"

#include "mlk/core/diagnostics.h"

#include <string>
#include "mlk/type/shape.h"
#include "mlk/core/constants.h"

namespace mlk {

namespace {

/// Broadcast two shapes; requires full compatibility.
bool broadcastShapes(const Shape& a, const Shape& b, Shape& out) {
    return Shape::broadcast(a, b, out);
}

}  // namespace

std::optional<MathType> inferResultType(const MathGraph& graph, NodeId nodeId,
                                        const MathDomainProfile* profile) {
    (void)profile;
    const Node& n = graph.node(nodeId);
    const MathType& t0 = graph.value(n.inputs[0]).type;

    auto sameDtypeError = [&]() -> std::optional<MathType> {
        for (const ValueId in : n.inputs) {
            const MathType& ti = graph.value(in).type;
            if (ti.dtype != t0.dtype) return std::nullopt;
        }
        return t0;
    };

    switch (n.op) {  // Rule 78: exhaustive
        case MathOp::Add: case MathOp::Sub: case MathOp::Mul:
        case MathOp::Div: case MathOp::Neg: case MathOp::Pow:
        case MathOp::Exp: case MathOp::Log: case MathOp::Sin:
        case MathOp::Cos: case MathOp::Tan: case MathOp::Tanh:
        case MathOp::Sqrt: case MathOp::Rsqrt: case MathOp::Erf:
        case MathOp::Gelu: {
            // Elementwise: shapes must broadcast; result dtype uniform.
            MathType out = t0;
            if (t0.tensor.has_value()) {
                Shape acc = t0.tensor->shape;
                for (std::size_t i = 1; i < n.inputs.size(); ++i) {
                    const MathType& ti = graph.value(n.inputs[i]).type;
                    if (!ti.tensor.has_value()) return std::nullopt;
                    Shape merged;
                    if (!broadcastShapes(acc, ti.tensor->shape, merged)) {
                        return std::nullopt;
                    }
                    acc = merged;
                }
                out.tensor->shape = acc;
                out.shape = acc;
            } else {
                for (std::size_t i = 1; i < n.inputs.size(); ++i) {
                    if (graph.value(n.inputs[i]).type != t0) {
                        return std::nullopt;
                    }
                }
            }
            return sameDtypeError() ? std::optional<MathType>{out}
                                    : std::nullopt;
        }
        case MathOp::MatMul: {
            // [m,k] x [k,n] -> [m,n]; non-commutative (Rule 33).
            const MathType& ta = graph.value(n.inputs[0]).type;
            const MathType& tb = graph.value(n.inputs[1]).type;
            if (!ta.tensor.has_value() || !tb.tensor.has_value()) {
                return std::nullopt;
            }
            if (ta.tensor->element != tb.tensor->element) return std::nullopt;
            const Shape& sa = ta.tensor->shape;
            const Shape& sb = tb.tensor->shape;
            if (sa.rank() != 2 || sb.rank() != 2) return std::nullopt;
            if (sa.dim(1) != sb.dim(0)) return std::nullopt;
            MathType out = ta;
            Shape res;
            res.addDim(sa.dim(0));
            res.addDim(sb.dim(1));
            out.shape = res;
            out.tensor = ta.tensor;
            out.tensor->shape = res;
            return out;
        }
        case MathOp::Dot: {
            const MathType& ta = graph.value(n.inputs[0]).type;
            if (!ta.tensor.has_value() || ta.tensor->shape.rank() != 1) {
                return std::nullopt;
            }
            MathType out = ta;
            out.shape = std::nullopt;
            out.tensor = std::nullopt;
            return out;
        }
        case MathOp::Transpose: {
            if (!t0.tensor.has_value() || t0.tensor->shape.rank() != 2) {
                return std::nullopt;
            }
            MathType out = t0;
            Shape res;
            res.addDim(t0.tensor->shape.dim(1));
            res.addDim(t0.tensor->shape.dim(0));
            out.shape = res;
            out.tensor = t0.tensor;
            out.tensor->shape = res;
            return out;
        }
        case MathOp::Reshape: {
            // Target shape arrives as a list attr of dims; MVP graphs only
            // reshape to static or fully-dynamic ranks (attrs are interned;
            // Rule 16).
            MathType out = t0;
            if (const AttrValue* v = findAttr(n.attrs, graph.wk().shapeDims)) {
                if (std::holds_alternative<int64_t>(v->v)) {
                    const int64_t rank = std::get<int64_t>(v->v);
                    if (rank >= 0 &&
                        rank <= static_cast<int64_t>(constants::kMaxRank)) {
                        Shape s;
                        for (int64_t i = 0; i < rank; ++i) s.addDim(kDynamicDim);
                        out.shape = s;
                        out.tensor = t0.tensor;
                        out.tensor->shape = s;
                    }
                }
            }
            return out;
        }
        case MathOp::ReduceSum: case MathOp::ReduceMax:
        case MathOp::ReduceMean: {
            if (!t0.tensor.has_value()) return std::nullopt;
            MathType out = t0;
            out.shape = std::nullopt;  // full reduction to scalar (axis attr
            out.tensor = std::nullopt;  // is a Tier-2 extension)
            return out;
        }
        case MathOp::Softmax: {
            return t0;  // same shape, elementwise-stable
        }
        case MathOp::Einsum: case MathOp::Conv: {
            // Lowered by tensor.einsum_lower / tensor passes before codegen;
            // the abstract op carries operand types through.
            return t0;
        }
        case MathOp::Derivative: case MathOp::Gradient: {
            // d f/d x has the same type as f (spec: calculus is first-class).
            return t0;
        }
        case MathOp::Integral: case MathOp::Limit: case MathOp::Solve: {
            return t0;
        }
        case MathOp::Apply: case MathOp::Lambda: {
            return t0;
        }
        case MathOp::IntToFloat: {
            MathType out = t0;
            out.domain = Domain::Float;
            out.dtype = Dtype::F64;
            return out;
        }
        case MathOp::FloatToInt: {
            MathType out = t0;
            out.domain = Domain::Int;
            out.dtype = Dtype::I64;
            return out;
        }
        case MathOp::RealToComplex: {
            MathType out = t0;
            out.domain = Domain::Complex;
            out.dtype = Dtype::C128;
            return out;
        }
        case MathOp::ScalarToTensor: {
            // Rule 39 explicit lift: the frontend annotates the broadcast
            // target type on the result value; inference preserves the
            // annotation when it is a well-formed tensor of the same
            // dtype, otherwise the node stays scalar-shaped (t0).
            const MathType& tin = graph.value(n.inputs[0]).type;
            const MathType annotated = graph.value(n.results[0]).type;
            if (annotated.tensor.has_value() && tin.isScalarLike() &&
                annotated.tensor->element == tin.dtype) {
                return annotated;
            }
            return t0;
        }
        case MathOp::TensorToScalar:
        case MathOp::LayoutTransform: case MathOp::Reinterpret:
        case MathOp::BitCast: case MathOp::Box: case MathOp::Unbox:
            return t0;
        case MathOp::NativeToMathRef: case MathOp::MathToNativeRef:
            return t0;
        case MathOp::Broadcast:  // broadcast result shape from attr or T0
        case MathOp::kCount:
            return t0;
    }
    return std::nullopt;
}

Result<uint32_t> inferTypes(MathGraph& graph,
                            const MathDomainProfile& profile) {
    uint32_t changed = 0;
    for (const NodeId nid : graph.topoOrder()) {
        Node& n = graph.node(nid);
        if (n.flags.test(NodeFlag::Dead)) continue;
        if (n.inputs.empty()) continue;
        auto inferred = inferResultType(graph, nid, &profile);
        if (!inferred.has_value()) {
            // Rule 67: actionable diagnostic — location, expected vs actual,
            // violated rule, suggested fix.
            Diagnostic d;
            d.severity = Severity::Error;
            d.nodeId = nid;
            d.message =
                std::string("type inference failed at '") + opName(n.op) +
                "' node";
            d.expected =
                "compatible operand dtypes/shapes (or explicit conversion)";
            d.actual = "operands require implicit coercion";
            d.rule = "Rule 39";
            d.suggestedFix =
                "insert an explicit conversion node (Rule 39 list) before "
                "this operation";
            return err(ErrorCode::InvalidGraph,
                       d.message + " (" + d.actual + "; " + d.rule + ": " +
                           d.suggestedFix + ")",
                       39);
        }
        Value& result = graph.value(n.results[0]);
        if (!(result.type == *inferred)) {
            result.type = *inferred;
            ++changed;
        }
    }
    graph.bumpVersion();
    return changed;
}

Result<uint32_t> inferShapes(MathGraph& graph) {
    // Shapes are assigned during inferTypes for tensor domains; this pass
    // exists as the standalone contract point (spec pipeline names both) and
    // validates static-shape integrity.
    uint32_t validated = 0;
    for (const auto& v : graph.values()) {
        if (v.type.shape.has_value() && v.type.tensor.has_value() &&
            v.type.shape->rank() != v.type.tensor->shape.rank()) {
            return err(ErrorCode::InvalidGraph,
                       "shape/tensor rank mismatch on value " +
                           std::to_string(v.id),
                       24);
        }
        ++validated;
    }
    return validated;
}

Result<uint32_t> inferTypesReported(MathGraph& graph,
                                    const MathDomainProfile& profile,
                                    DiagnosticEngine& diag) {
    auto result = inferTypes(graph, profile);
    if (!result.has_value()) {
        Diagnostic d;
        d.severity = Severity::Error;
        d.message = result.error().message;
        d.rule = "Rule 39";
        d.suggestedFix =
            "insert an explicit conversion node (Rule 39 list) before the "
            "failing operation, or make operand dtypes/shapes compatible";
        diag.report(std::move(d));
    }
    return result;
}

}  // namespace mlk