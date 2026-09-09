// Math Domain Profile JSON + capability names.
// Loader validates untrusted input (Rule 124) and is the only place profiles
// are materialized (Part 0: profiles are the domain boundary).
#include "mlk/type/domain_profile.h"

#include "mlk/support/json.h"

namespace mlk {

const char* capabilityName(Capability c) noexcept {
    switch (c) {  // Rule 78: exhaustive
        case Capability::HasSymbolicValues: return "HasSymbolicValues";
        case Capability::HasNumericValues: return "HasNumericValues";
        case Capability::HasExactArithmetic: return "HasExactArithmetic";
        case Capability::HasFloatingPoint: return "HasFloatingPoint";
        case Capability::HasComplexNumbers: return "HasComplexNumbers";
        case Capability::HasArbitraryPrecision: return "HasArbitraryPrecision";
        case Capability::HasIntervals: return "HasIntervals";
        case Capability::HasFunctionValues: return "HasFunctionValues";
        case Capability::HasOperators: return "HasOperators";
        case Capability::HasTensorDomain: return "HasTensorDomain";
        case Capability::HasSparseDomain: return "HasSparseDomain";
        case Capability::HasDynamicShapes: return "HasDynamicShapes";
        case Capability::HasSymbolicShapes: return "HasSymbolicShapes";
        case Capability::HasCalculus: return "HasCalculus";
        case Capability::HasDerivatives: return "HasDerivatives";
        case Capability::HasIntegrals: return "HasIntegrals";
        case Capability::HasLimits: return "HasLimits";
        case Capability::HasGradients: return "HasGradients";
        case Capability::HasSolvers: return "HasSolvers";
        case Capability::HasAlgebraicRewriting: return "HasAlgebraicRewriting";
        case Capability::HasApproximation: return "HasApproximation";
        case Capability::HasULPContracts: return "HasULPContracts";
        case Capability::HasFastMath: return "HasFastMath";
        case Capability::HasErrorBounds: return "HasErrorBounds";
        case Capability::HasLayoutTransforms: return "HasLayoutTransforms";
        case Capability::HasKernelFusion: return "HasKernelFusion";
        case Capability::HasAutotuning: return "HasAutotuning";
        case Capability::HasSuperoptimization: return "HasSuperoptimization";
        case Capability::HasRuntimeCodeGeneration:
            return "HasRuntimeCodeGeneration";
        case Capability::HasDynamicGraphMutation:
            return "HasDynamicGraphMutation";
        case Capability::HasPhysicalMemoryPlacement:
            return "HasPhysicalMemoryPlacement";
        case Capability::HasCPUBackend: return "HasCPUBackend";
        case Capability::HasGPUBackend: return "HasGPUBackend";
        case Capability::HasAcceleratorBackend: return "HasAcceleratorBackend";
        case Capability::HasBenchmarkFeedback: return "HasBenchmarkFeedback";
        case Capability::HasHardwareCounters: return "HasHardwareCounters";
        case Capability::HasProofCertificates: return "HasProofCertificates";
        case Capability::HasStaticCertification: return "HasStaticCertification";
        case Capability::HasTracingHooks: return "HasTracingHooks";
        case Capability::HasProfilerHooks: return "HasProfilerHooks";
        case Capability::HasDebuggerHooks: return "HasDebuggerHooks";
        case Capability::HasFFI: return "HasFFI";
        case Capability::HasGlobalRuntimeLock: return "HasGlobalRuntimeLock";
        case Capability::HasFreeThreading: return "HasFreeThreading";
        case Capability::HasAsyncKernels: return "HasAsyncKernels";
        case Capability::HasStreamingEvaluation:
            return "HasStreamingEvaluation";
        case Capability::HasResumableEvaluation:
            return "HasResumableEvaluation";
        case Capability::kCount: return "?";
    }
    return "?";
}

bool capabilityByName(const char* name, Capability& out) noexcept {
    for (uint32_t i = 0; i < static_cast<uint32_t>(Capability::kCount); ++i) {
        const auto c = static_cast<Capability>(i);
        if (__builtin_strcmp(capabilityName(c), name) == 0) {
            out = c;
            return true;
        }
    }
    return false;
}

namespace {

bool parseTriState(const json::Value* v, TriState& out) {
    if (v == nullptr || !v->isString()) return false;
    if (v->asString() == "true") out = TriState::True;
    else if (v->asString() == "false") out = TriState::False;
    else if (v->asString() == "unknown") out = TriState::Unknown;
    else return false;
    return true;
}

}  // namespace

Result<MathDomainProfile> domainProfileFromJson(const json::Value& doc,
                                                SymbolTable& symbols) {
    MathDomainProfile p;
    const json::Value* name = doc.find("name");
    if (name == nullptr || !name->isString()) {
        return err(ErrorCode::ParseError, "profile missing name");
    }
    p.name = symbols.intern(name->asString());

    const json::Value* version = doc.find("version");
    if (version != nullptr && version->isString()) {
        p.version = symbols.intern(version->asString());
    }

    if (const json::Value* caps = doc.find("capabilities")) {
        if (!caps->isArray()) {
            return err(ErrorCode::ParseError, "capabilities must be an array");
        }
        for (const auto& c : caps->asArray()) {
            if (!c.isString()) {
                return err(ErrorCode::ParseError,
                           "capability entries must be strings");
            }
            Capability cap;
            if (!capabilityByName(c.asString().c_str(), cap)) {
                return err(ErrorCode::ParseError,
                           "unknown capability: " + c.asString());
            }
            p.capabilities.set(cap);
        }
    }

    if (const json::Value* num = doc.find("numeric")) {
        if (const json::Value* v = num->find("preserve_nan")) {
            if (v->isBool()) p.numeric.preserveNaN = v->asBool();
        }
        if (const json::Value* v = num->find("preserve_negative_zero")) {
            if (v->isBool()) p.numeric.preserveNegativeZero = v->asBool();
        }
        if (const json::Value* v = num->find("allow_reassociation")) {
            if (v->isBool()) p.numeric.allowReassociation = v->asBool();
        }
        if (const json::Value* v = num->find("allow_fma_contraction")) {
            if (v->isBool()) p.numeric.allowFMAContraction = v->asBool();
        }
        if (const json::Value* v = num->find("allow_approx_reciprocal")) {
            if (v->isBool()) p.numeric.allowApproximateReciprocal = v->asBool();
        }
    }

    if (const json::Value* approx = doc.find("approximation")) {
        if (const json::Value* v = approx->find("allow_approximation")) {
            if (v->isBool()) p.approximation.allowApproximation = v->asBool();
        }
        if (const json::Value* v = approx->find("default_max_ulps")) {
            if (v->isNumber()) p.approximation.defaultMaxUlps = v->asDouble();
        }
        if (const json::Value* v = approx->find("default_max_abs_error")) {
            if (v->isNumber()) {
                p.approximation.defaultMaxAbsError = v->asDouble();
            }
        }
        if (const json::Value* v = approx->find("default_max_rel_error")) {
            if (v->isNumber()) {
                p.approximation.defaultMaxRelError = v->asDouble();
            }
        }
    }

    if (const json::Value* laws = doc.find("laws")) {
        if (!parseTriState(laws->find("commutative_mul"),
                           p.lawCommutativeMul)) {
            return err(ErrorCode::ParseError,
                       "laws.commutative_mul must be tri-state string");
        }
        if (!parseTriState(laws->find("associative_add"),
                           p.lawAssociativeAdd)) {
            return err(ErrorCode::ParseError,
                       "laws.associative_add must be tri-state string");
        }
        if (!parseTriState(laws->find("distributive_mul_add"),
                           p.lawDistributiveMulAdd)) {
            return err(ErrorCode::ParseError,
                       "laws.distributive_mul_add must be tri-state string");
        }
    }

    p.dialectHash = static_cast<uint32_t>(p.hash() & 0xFFFFFFFFu);
    return p;
}

json::Value domainProfileToJson(const MathDomainProfile& p,
                                SymbolTable& symbols) {
    json::Value doc = json::Object{};
    doc.set("name", json::Value{symbols.text(p.name)});
    if (p.version != kInvalidSymbolId) {
        doc.set("version", json::Value{symbols.text(p.version)});
    }
    json::Value caps = json::Array{};
    for (uint32_t i = 0; i < static_cast<uint32_t>(Capability::kCount); ++i) {
        const auto c = static_cast<Capability>(i);
        if (p.capabilities.test(c)) caps.push(json::Value{capabilityName(c)});
    }
    doc.set("capabilities", std::move(caps));

    json::Value num = json::Object{};
    num.set("preserve_nan", json::Value{p.numeric.preserveNaN});
    num.set("preserve_negative_zero", json::Value{p.numeric.preserveNegativeZero});
    num.set("preserve_inf", json::Value{p.numeric.preserveInf});
    num.set("allow_reassociation", json::Value{p.numeric.allowReassociation});
    num.set("allow_fma_contraction", json::Value{p.numeric.allowFMAContraction});
    num.set("allow_approx_reciprocal",
            json::Value{p.numeric.allowApproximateReciprocal});
    doc.set("numeric", std::move(num));

    json::Value approx = json::Object{};
    approx.set("allow_approximation",
               json::Value{p.approximation.allowApproximation});
    approx.set("default_max_ulps", json::Value{p.approximation.defaultMaxUlps});
    approx.set("default_max_abs_error",
               json::Value{p.approximation.defaultMaxAbsError});
    approx.set("default_max_rel_error",
               json::Value{p.approximation.defaultMaxRelError});
    doc.set("approximation", std::move(approx));

    auto triStr = [](TriState t) {
        return t == TriState::True ? "true"
               : t == TriState::False ? "false" : "unknown";
    };
    json::Value laws = json::Object{};
    laws.set("commutative_mul", json::Value{triStr(p.lawCommutativeMul)});
    laws.set("associative_add", json::Value{triStr(p.lawAssociativeAdd)});
    laws.set("distributive_mul_add",
             json::Value{triStr(p.lawDistributiveMulAdd)});
    doc.set("laws", std::move(laws));
    return doc;
}

}  // namespace mlk
