// Graph JSON (de)serialization. Format version kGraphFileFormatVersion.
// The loader is the trust boundary for all .mlk inputs (Rule 124: malformed
// inputs are rejected with actionable diagnostics, never UB).
#include "mlk/ir/graph_json.h"

#include <cstdio>

namespace mlk {

namespace {

json::Value shapeToJson(const Shape& s) {
    json::Value arr = json::Array{};
    for (std::size_t i = 0; i < s.rank(); ++i) {
        arr.push(json::Value{s.dim(i)});
    }
    return arr;
}

Result<Shape> shapeFromJson(const json::Value& v) {
    if (!v.isArray()) return err(ErrorCode::ParseError, "shape must be array");
    Shape s;
    for (const auto& d : v.asArray()) {
        if (!d.isInt()) return err(ErrorCode::ParseError, "dim must be int");
        s.addDim(d.asInt());
    }
    return s;
}

json::Value factsToJson(const FactSet& facts) {
    json::Value arr = json::Array{};
    for (const auto& f : facts.all()) {
        json::Value fo = json::Object{};
        fo.set("property", json::Value{propertyName(f.property)});
        const char* tv = f.value == TriState::True
                             ? "true"
                             : (f.value == TriState::False ? "false" : "unknown");
        fo.set("value", json::Value{tv});
        if (f.property == PropertyId::Bounded) {
            fo.set("range_low", json::Value{f.range.low});
            fo.set("range_high", json::Value{f.range.high});
        }
        if (f.property == PropertyId::Periodic) {
            fo.set("period", json::Value{f.period});
        }
        arr.push(std::move(fo));
    }
    return arr;
}

}  // namespace

json::Value graphToJson(const MathGraph& graph, SymbolTable& symbols) {
    json::Value doc = json::Object{};
    doc.set("format", json::Value{"mlk-graph"});
    doc.set("version",
            json::Value{static_cast<int64_t>(constants::kGraphFileFormatVersion)});

    json::Value vals = json::Array{};
    for (const auto& v : graph.values()) {
        json::Value vo = json::Object{};
        vo.set("id", json::Value{static_cast<int64_t>(v.id)});
        vo.set("kind", json::Value{valueKindName(v.kind)});
        if (v.name != kInvalidSymbolId) {
            vo.set("name", json::Value{symbols.text(v.name)});
        }
        json::Value t = json::Object{};
        t.set("domain", json::Value{domainName(v.type.domain)});
        t.set("dtype", json::Value{dtypeName(v.type.dtype)});
        t.set("algebra", json::Value{algebraicClassName(v.type.algebra)});
        t.set("diff", json::Value{differentiabilityName(v.type.diff)});
        if (v.type.shape) t.set("shape", shapeToJson(*v.type.shape));
        vo.set("type", std::move(t));
        if (v.kind == ValueKind::Constant) {
            if (v.constant.isInt) {
                vo.set("int_value", json::Value{v.constant.i64});
            } else {
                vo.set("float_value", json::Value{v.constant.f64});
            }
        }
        if (!v.facts.empty()) vo.set("facts", factsToJson(v.facts));
        vals.push(std::move(vo));
    }
    doc.set("values", std::move(vals));

    json::Value nodes = json::Array{};
    for (const auto& n : graph.nodes()) {
        json::Value no = json::Object{};
        no.set("id", json::Value{static_cast<int64_t>(n.id)});
        no.set("op", json::Value{opName(n.op)});
        json::Value ins = json::Array{};
        for (const ValueId in : n.inputs) {
            ins.push(json::Value{static_cast<int64_t>(in)});
        }
        no.set("inputs", std::move(ins));
        no.set("dead", json::Value{n.flags.test(NodeFlag::Dead)});
        nodes.push(std::move(no));
    }
    doc.set("nodes", std::move(nodes));

    json::Value outs = json::Array{};
    for (const ValueId v : graph.outputs()) {
        outs.push(json::Value{static_cast<int64_t>(v)});
    }
    doc.set("outputs", std::move(outs));
    return doc;
}

Result<MathGraph> graphFromJson(const json::Value& doc, SymbolTable& symbols) {
    const json::Value* format = doc.find("format");
    if (format == nullptr || !format->isString() ||
        format->asString() != "mlk-graph") {
        return err(ErrorCode::ParseError, "not an mlk-graph document");
    }
    const json::Value* version = doc.find("version");
    if (version == nullptr || !version->isInt() ||
        version->asInt() != static_cast<int64_t>(
                                constants::kGraphFileFormatVersion)) {
        return err(ErrorCode::ParseError,
                   "unsupported graph format version (expected " +
                       std::to_string(constants::kGraphFileFormatVersion) +
                       ")");
    }

    MathGraph graph{&symbols};

    const json::Value* vals = doc.find("values");
    if (vals == nullptr || !vals->isArray()) {
        return err(ErrorCode::ParseError, "missing values array");
    }
    // Values must come back with their original ids to keep operand refs
    // valid; ids are dense and must match file order.
    uint32_t expectedId = 0;
    for (const auto& vo : vals->asArray()) {
        const json::Value* idv = vo.find("id");
        if (idv == nullptr || !idv->isInt() ||
            idv->asInt() != static_cast<int64_t>(expectedId)) {
            return err(ErrorCode::ParseError,
                       "value ids must be dense in file order");
        }
        ++expectedId;
        const json::Value* kindv = vo.find("kind");
        if (kindv == nullptr || !kindv->isString()) {
            return err(ErrorCode::ParseError, "value missing kind");
        }
        const std::string kind = kindv->asString();

        MathType type;
        if (const json::Value* tv = vo.find("type")) {
            if (const json::Value* d = tv->find("domain")) {
                if (d->isString()) {
                    // Dense domain parse via name table.
                    const char* dn = d->asString().c_str();
                    for (uint8_t i = 0; i <= static_cast<uint8_t>(Domain::Distribution);
                         ++i) {
                        if (__builtin_strcmp(domainName(static_cast<Domain>(i)),
                                             dn) == 0) {
                            type.domain = static_cast<Domain>(i);
                            break;
                        }
                    }
                }
            }
            if (const json::Value* dt = tv->find("dtype")) {
                if (dt->isString()) {
                    const char* dtn = dt->asString().c_str();
                    for (uint8_t i = 0; i <= static_cast<uint8_t>(Dtype::C128);
                         ++i) {
                        if (__builtin_strcmp(dtypeName(static_cast<Dtype>(i)),
                                             dtn) == 0) {
                            type.dtype = static_cast<Dtype>(i);
                            break;
                        }
                    }
                }
            }
            if (const json::Value* sh = tv->find("shape")) {
                Shape parsedShape;
                MLK_TRY(parsedShape, shapeFromJson(*sh));
                type.shape = parsedShape;
            }
        }

        const json::Value* namev = vo.find("name");
        SymbolId name = kInvalidSymbolId;
        if (namev != nullptr && namev->isString()) {
            name = symbols.intern(namev->asString());
        }

        ValueId vid = kInvalidValueId;
        if (kind == "placeholder") {
            vid = graph.addPlaceholder(name, type);
        } else if (kind == "variable") {
            vid = graph.addVariable(name, type);
        } else if (kind == "symbol") {
            vid = graph.addSymbol(name, type);
        } else if (kind == "constant") {
            if (const json::Value* iv = vo.find("int_value"); iv && iv->isInt()) {
                vid = graph.addIntConstant(iv->asInt(), type);
            } else if (const json::Value* fv = vo.find("float_value");
                       fv && fv->isNumber()) {
                vid = graph.addConstant(fv->asDouble(), type);
            } else {
                return err(ErrorCode::ParseError,
                           "constant missing numeric payload");
            }
        } else {
            return err(ErrorCode::ParseError, "unknown value kind: " + kind);
        }
        (void)vid;
    }

    const json::Value* nodes = doc.find("nodes");
    if (nodes == nullptr || !nodes->isArray()) {
        return err(ErrorCode::ParseError, "missing nodes array");
    }
    for (const auto& no : nodes->asArray()) {
        const json::Value* opv = no.find("op");
        if (opv == nullptr || !opv->isString()) {
            return err(ErrorCode::ParseError, "node missing op");
        }
        MathOp op;
        if (!opByName(opv->asString().c_str(), op)) {
            return err(ErrorCode::ParseError, "unknown op: " + opv->asString());
        }
        const json::Value* ins = no.find("inputs");
        if (ins == nullptr || !ins->isArray()) {
            return err(ErrorCode::ParseError, "node missing inputs");
        }
        SmallVector<ValueId, 4> inputs;
        for (const auto& i : ins->asArray()) {
            if (!i.isInt()) return err(ErrorCode::ParseError, "input not int");
            const int64_t iv = i.asInt();
            if (iv < 0 || iv >= static_cast<int64_t>(graph.numValues())) {
                return err(ErrorCode::ParseError, "input id out of range");
            }
            inputs.push_back(static_cast<ValueId>(iv));
        }
        MLK_TRYV(graph.addNode(op, inputs));
    }

    if (const json::Value* outs = doc.find("outputs")) {
        if (!outs->isArray()) {
            return err(ErrorCode::ParseError, "outputs must be array");
        }
        for (const auto& o : outs->asArray()) {
            if (!o.isInt() || o.asInt() < 0 ||
                o.asInt() >= static_cast<int64_t>(graph.numValues())) {
                return err(ErrorCode::ParseError, "output id out of range");
            }
            graph.addOutput(static_cast<ValueId>(o.asInt()));
        }
    }
    return graph;
}

}  // namespace mlk
