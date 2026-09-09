// Graph printer implementation.
#include "mlk/ir/graph_printer.h"

#include <cstdio>

namespace mlk {

namespace {

void appendValue(const MathGraph& graph, ValueId vid, SymbolTable& symbols,
                 const PrintOptions& opts, std::string& out);

void appendOp(const MathGraph& graph, NodeId nid, SymbolTable& symbols,
              const PrintOptions& opts, std::string& out) {
    const Node& n = graph.node(nid);
    if (!opts.printDead && n.flags.test(NodeFlag::Dead)) {
        out += "/*dead*/";
    }
    out.push_back('(');
    out += opName(n.op);
    for (const ValueId in : n.inputs) {
        out.push_back(' ');
        appendValue(graph, in, symbols, opts, out);
    }
    for (const auto& a : n.attrs) {
        out.push_back(' ');
        out += ':' + symbols.text(a.name) + '=';
        if (std::holds_alternative<int64_t>(a.value.v)) {
            out += std::to_string(std::get<int64_t>(a.value.v));
        } else if (std::holds_alternative<double>(a.value.v)) {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%g", std::get<double>(a.value.v));
            out += buf;
        } else if (std::holds_alternative<bool>(a.value.v)) {
            out += std::get<bool>(a.value.v) ? "true" : "false";
        } else {
            out += '@' + symbols.text(std::get<SymbolId>(a.value.v));
        }
    }
    out.push_back(')');
}

void appendValue(const MathGraph& graph, ValueId vid, SymbolTable& symbols,
                 const PrintOptions& opts, std::string& out) {
    const Value& v = graph.value(vid);
    switch (v.kind) {  // Rule 78: exhaustive
        case ValueKind::Constant:
            if (v.constant.isInt) {
                out += std::to_string(v.constant.i64);
            } else {
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%g", v.constant.f64);
                out += buf;
            }
            break;
        case ValueKind::Variable:
        case ValueKind::Placeholder:
        case ValueKind::Symbol:
            out += v.name == kInvalidSymbolId ? "_"
                                              : symbols.text(v.name);
            break;
        case ValueKind::NodeResult:
            if (v.producer != kInvalidNodeId) {
                appendOp(graph, v.producer, symbols, opts, out);
            } else {
                out += "_";
            }
            break;
    }
    if (opts.annotateTypes) {
        out.push_back(':');
        out += domainName(v.type.domain);
        if (v.type.shape) {
            out.push_back('[');
            for (std::size_t i = 0; i < v.type.shape->rank(); ++i) {
                if (i != 0) out.push_back(',');
                const int64_t d = v.type.shape->dim(i);
                out += d == kDynamicDim ? "?" : std::to_string(d);
            }
            out.push_back(']');
        }
    }
}

}  // namespace

std::string printGraph(const MathGraph& graph, const PrintOptions& opts,
                       SymbolTable& symbols) {
    std::string out;
    out += "(graph";
    if (!graph.outputs().empty()) {
        out += " (outputs";
        for (const ValueId v : graph.outputs()) {
            out.push_back(' ');
            appendValue(graph, v, symbols, opts, out);
        }
        out.push_back(')');
    }
    if (opts.printDead) {
        for (const auto& n : graph.nodes()) {
            if (n.flags.test(NodeFlag::Dead)) {
                out += " (dead ";
                appendOp(graph, n.id, symbols, opts, out);
                out.push_back(')');
            }
        }
    }
    out.push_back(')');
    return out;
}

std::string printValueExpr(const MathGraph& graph, ValueId v,
                           SymbolTable& symbols) {
    std::string out;
    PrintOptions opts;
    appendValue(graph, v, symbols, opts, out);
    return out;
}

}  // namespace mlk
