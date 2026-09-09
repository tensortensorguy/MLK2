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
    }
    for (const auto& n : nodes) {
        h = hashCombine(h, hashU64(static_cast<uint64_t>(n.op)));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(n.math)));
        h = hashCombine(h, hashI64(n.begin));
        h = hashCombine(h, hashI64(n.end));
        h = hashCombine(h, hashI64(n.step));
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
        bufs.push(std::move(bo));
    }
    doc.set("buffers", std::move(bufs));

    json::Value ns = json::Array{};
    for (const auto& n : nodes) {
        json::Value no = json::Object{};
        no.set("op", json::Value{kernelOpName(n.op)});
        if (n.op == KernelOp::Compute) {
            no.set("math", json::Value{opName(n.math)});
        }
        if (n.var != kInvalidSymbolId) {
            no.set("var", json::Value{symbols.text(n.var)});
        }
        no.set("begin", json::Value{n.begin});
        no.set("end", json::Value{n.end});
        no.set("step", json::Value{n.step});
        if (n.bufferA != constants::kInvalidId) {
            no.set("a", json::Value{static_cast<int64_t>(n.bufferA)});
        }
        if (n.bufferB != constants::kInvalidId) {
            no.set("b", json::Value{static_cast<int64_t>(n.bufferB)});
        }
        if (n.bufferOut != constants::kInvalidId) {
            no.set("out", json::Value{static_cast<int64_t>(n.bufferOut)});
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
