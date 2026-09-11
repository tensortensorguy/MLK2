// KernelIR implementation (hash/serialization/names).
#include "mlk/support/kernel_ir.h"

namespace mlk {

const char* kernelOpName(KernelOp op) noexcept {
    switch (op) {  // Rule 78: exhaustive
        case KernelOp::Loop: return "loop";
        case KernelOp::Compute: return "compute";
        case KernelOp::Load: return "load";
        case KernelOp::Store: return "store";
        case KernelOp::AllocBuffer: return "alloc";
        case KernelOp::CopyBuffer: return "copy";
        case KernelOp::Barrier: return "barrier";
        case KernelOp::Trace: return "trace";
        case KernelOp::Guard: return "guard";
        case KernelOp::Call: return "call";
        case KernelOp::kCount: return "?";
    }
    return "?";
}

bool isScalarRealizable(MathOp op) noexcept {
    switch (op) {  // Rule 78: exhaustive
        case MathOp::Add: case MathOp::Sub: case MathOp::Mul:
        case MathOp::Div: case MathOp::Neg: case MathOp::Pow:
        case MathOp::Exp:
        case MathOp::Log: case MathOp::Sin: case MathOp::Cos:
        case MathOp::Tan: case MathOp::Tanh: case MathOp::Sqrt:
        case MathOp::Rsqrt: case MathOp::Erf: case MathOp::Gelu:
            return true;
        default:
            return false;
    }
}

HashValue KernelModule::hash() const noexcept {
    HashValue h = kHashSeed;
    for (const auto& b : buffers) {
        h = hashCombine(h, hashU64(b.name));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(b.dtype)));
        h = hashCombine(h, hashI64(b.elements));
        for (const int64_t d : b.dims) h = hashCombine(h, hashI64(d));
    }
    for (const auto& n : nodes) {
        h = hashCombine(h, hashU64(static_cast<uint64_t>(n.op)));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(n.math)));
        h = hashCombine(h, hashI64(n.begin));
        h = hashCombine(h, hashI64(n.end));
        h = hashCombine(h, hashI64(n.step));
        h = hashCombine(h, hashU64(n.endBuf));
        h = hashCombine(h, hashI64(n.endDim));
        for (const int64_t v : n.beginCoeffs) h = hashCombine(h, hashI64(v));
        h = hashCombine(h, hashI64(n.beginOffset));
        for (const int64_t v : n.endCoeffs) h = hashCombine(h, hashI64(v));
        h = hashCombine(h, hashI64(n.endOffset));
        h = hashCombine(h, hashU64(n.family));
        for (const int64_t c : n.outIndexCoeffs) h = hashCombine(h, hashI64(c));
        h = hashCombine(h, hashI64(n.outIndexOffset));
        h = hashCombine(h, n.accumulate ? 0x9E3779B97F4A7C15ULL : 0ULL);
        for (const auto& e : n.exprs) {
            h = hashCombine(h, hashU64(static_cast<uint64_t>(e.op)));
            h = hashCombine(h, hashU64(static_cast<uint64_t>(e.a.kind)));
            h = hashCombine(h, hashI64(e.a.index));
            for (const int64_t c : e.a.idxCoeffs) {
                h = hashCombine(h, hashI64(c));
            }
            h = hashCombine(h, hashI64(e.a.idxOffset));
            h = hashCombine(h, hashU64(static_cast<uint64_t>(e.b.kind)));
            h = hashCombine(h, hashI64(e.b.index));
            for (const int64_t c : e.b.idxCoeffs) {
                h = hashCombine(h, hashI64(c));
            }
            h = hashCombine(h, hashI64(e.b.idxOffset));
            // Const payloads hash their bit pattern (Rule 24: stable).
            std::uint64_t abits = 0, bbits = 0;
            if (e.a.kind == KernelOperand::Kind::Const) {
                double ad = e.a.constValue;
                __builtin_memcpy(&abits, &ad, sizeof(abits));
            }
            if (e.b.kind == KernelOperand::Kind::Const) {
                double bd = e.b.constValue;
                __builtin_memcpy(&bbits, &bd, sizeof(bbits));
            }
            h = hashCombine(h, abits);
            h = hashCombine(h, bbits);
        }
    }
    return h;
}

json::Value KernelModule::toJson(SymbolTable& symbols) const {
    json::Value doc = json::Object{};
    doc.set("format", json::Value{"mlk-kernel"});
    doc.set("version",
            json::Value{static_cast<int64_t>(
                constants::kBytecodeFormatVersion)});
    if (name != kInvalidSymbolId) {
        doc.set("name", json::Value{symbols.text(name)});
    }
    json::Value bufs = json::Array{};
    for (const auto& b : buffers) {
        json::Value bo = json::Object{};
        bo.set("name", json::Value{symbols.text(b.name)});
        bo.set("dtype", json::Value{dtypeName(b.dtype)});
        bo.set("input", json::Value{b.isInput});
        bo.set("output", json::Value{b.isOutput});
        bo.set("elements", json::Value{b.elements});
        if (!b.dims.empty()) {
            json::Value ds = json::Array{};
            for (const int64_t d : b.dims) {
                ds.push(json::Value{d});
            }
            bo.set("dims", std::move(ds));
        }
        bufs.push(std::move(bo));
    }
    doc.set("buffers", std::move(bufs));

    json::Value ns = json::Array{};
    for (const auto& n : nodes) {
        json::Value no = json::Object{};
        no.set("op", json::Value{kernelOpName(n.op)});
        if (n.op == KernelOp::Compute) {
            no.set("math", json::Value{opName(n.math)});
            if (n.family != kInvalidSymbolId) {
                no.set("family", json::Value{symbols.text(n.family)});
            }
        }
        if (n.var != kInvalidSymbolId) {
            no.set("var", json::Value{symbols.text(n.var)});
        }
        no.set("begin", json::Value{n.begin});
        no.set("end", json::Value{n.end});
        no.set("step", json::Value{n.step});
        if (n.endBuf != constants::kInvalidId) {
            no.set("end_buf", json::Value{static_cast<int64_t>(n.endBuf)});
            no.set("end_dim", json::Value{static_cast<int64_t>(n.endDim)});
        }
        if (!n.beginCoeffs.empty() || !n.endCoeffs.empty()) {
            json::Value bc = json::Array{};
            for (const int64_t v : n.beginCoeffs) bc.push(json::Value{v});
            no.set("begin_coeffs", std::move(bc));
            no.set("begin_offset", json::Value{n.beginOffset});
            json::Value ec = json::Array{};
            for (const int64_t v : n.endCoeffs) ec.push(json::Value{v});
            no.set("end_coeffs", std::move(ec));
            no.set("end_offset", json::Value{n.endOffset});
        }
        if (n.hasAffineStore()) {
            json::Value cs = json::Array{};
            for (const int64_t c : n.outIndexCoeffs) {
                cs.push(json::Value{c});
            }
            no.set("out_index_coeffs", std::move(cs));
            no.set("out_index_offset", json::Value{n.outIndexOffset});
            if (n.accumulate) no.set("accumulate", json::Value{true});
        }
        if (n.bufferA != constants::kInvalidId) {
            no.set("a", json::Value{static_cast<int64_t>(n.bufferA)});
        }
        if (n.bufferB != constants::kInvalidId) {
            no.set("b", json::Value{static_cast<int64_t>(n.bufferB)});
        }
        if (n.bufferOut != constants::kInvalidId) {
            no.set("out", json::Value{static_cast<int64_t>(n.bufferOut)});
        }
        if (!n.exprs.empty()) {
            json::Value xs = json::Array{};
            for (const auto& e : n.exprs) {
                json::Value eo = json::Object{};
                eo.set("op", json::Value{opName(e.op)});
                auto operandJson = [&](const KernelOperand& o,
                                       const char* slot) {
                    json::Value oo = json::Object{};
                    const char* k = "const";
                    switch (o.kind) {  // Rule 78: exhaustive
                        case KernelOperand::Kind::Const: k = "const"; break;
                        case KernelOperand::Kind::ElemA: k = "elem_a"; break;
                        case KernelOperand::Kind::ElemB: k = "elem_b"; break;
                        case KernelOperand::Kind::ScalarParam:
                            k = "scalar_param";
                            break;
                        case KernelOperand::Kind::Temp: k = "temp"; break;
                        case KernelOperand::Kind::ElemIdx:
                            k = "elem_idx";
                            break;
                    }
                    oo.set("kind", json::Value{k});
                    if (o.kind == KernelOperand::Kind::Const) {
                        oo.set("value", json::Value{o.constValue});
                    }
                    if (o.kind == KernelOperand::Kind::Temp ||
                        o.kind == KernelOperand::Kind::ScalarParam) {
                        oo.set("index", json::Value{o.index});
                    }
                    if (o.kind == KernelOperand::Kind::ElemIdx) {
                        oo.set("buffer", json::Value{o.index});
                        json::Value cs = json::Array{};
                        for (const int64_t c : o.idxCoeffs) {
                            cs.push(json::Value{c});
                        }
                        oo.set("coeffs", std::move(cs));
                        oo.set("offset", json::Value{o.idxOffset});
                    }
                    eo.set(slot, std::move(oo));
                };
                operandJson(e.a, "a");
                operandJson(e.b, "b");
                xs.push(std::move(eo));
            }
            no.set("exprs", std::move(xs));
        }
        json::Value kids = json::Array{};
        for (const uint32_t c : n.children) {
            kids.push(json::Value{static_cast<int64_t>(c)});
        }
        no.set("children", std::move(kids));
        ns.push(std::move(no));
    }
    doc.set("nodes", std::move(ns));

    json::Value sched = json::Object{};
    scheduleParams.forEach([&](SymbolId k, int64_t v) {
        sched.set(symbols.text(k), json::Value{v});
    });
    doc.set("schedule", std::move(sched));
    return doc;
}

}  // namespace mlk
