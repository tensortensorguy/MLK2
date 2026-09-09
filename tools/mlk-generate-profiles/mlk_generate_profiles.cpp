// mlk-generate-profiles — emits Math Domain Profile JSONs into profiles/
// (Part 0: profiles are versioned, reviewable artifacts).
#include <cstdio>
#include <cstring>

#include "mlk/core/symbol_table.h"
#include "mlk/type/domain_profile.h"

namespace {
void writeProfile(const char* dir, const char* name,
                  mlk::CapabilityFlags caps, bool approx,
                  mlk::TriState assocAdd) {
    mlk::SymbolTable symbols;
    mlk::MathDomainProfile p;
    p.name = symbols.intern(name);
    p.version = symbols.intern("1.0.0");
    p.capabilities = caps;
    p.approximation.allowApproximation = approx;
    p.lawCommutativeMul = mlk::TriState::True;
    p.lawAssociativeAdd = assocAdd;
    p.lawDistributiveMulAdd = mlk::TriState::True;
    mlk::json::Value doc = mlk::domainProfileToJson(p, symbols);
    std::string path = std::string(dir) + "/profile.json";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "mlk-generate-profiles: cannot write %s\n",
                     path.c_str());
        return;
    }
    const std::string text = mlk::json::serializePretty(doc);
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    std::printf("wrote %s\n", path.c_str());
}
}  // namespace

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "profiles";

    mlk::CapabilityFlags scalarCaps;
    scalarCaps.set(mlk::Capability::HasNumericValues);
    scalarCaps.set(mlk::Capability::HasFloatingPoint);
    scalarCaps.set(mlk::Capability::HasExactArithmetic);
    scalarCaps.set(mlk::Capability::HasAlgebraicRewriting);
    scalarCaps.set(mlk::Capability::HasCPUBackend);
    writeProfile((std::string(dir) + "/scalar_f64").c_str(), "scalar_f64",
                 scalarCaps, false, mlk::TriState::Unknown);

    mlk::CapabilityFlags tensorCaps;
    tensorCaps.set(mlk::Capability::HasNumericValues);
    tensorCaps.set(mlk::Capability::HasFloatingPoint);
    tensorCaps.set(mlk::Capability::HasTensorDomain);
    tensorCaps.set(mlk::Capability::HasDynamicShapes);
    tensorCaps.set(mlk::Capability::HasLayoutTransforms);
    tensorCaps.set(mlk::Capability::HasKernelFusion);
    tensorCaps.set(mlk::Capability::HasAutotuning);
    tensorCaps.set(mlk::Capability::HasCPUBackend);
    writeProfile((std::string(dir) + "/tensor_f32_cpu").c_str(),
                 "tensor_f32_cpu", tensorCaps, false, mlk::TriState::Unknown);

    mlk::CapabilityFlags gpuCaps = tensorCaps;
    gpuCaps.set(mlk::Capability::HasGPUBackend);
    gpuCaps.set(mlk::Capability::HasAsyncKernels);
    // f16 on GPU: no CPU backend executes it — backend capability only.
    writeProfile((std::string(dir) + "/tensor_f16_gpu").c_str(),
                 "tensor_f16_gpu", gpuCaps, false, mlk::TriState::Unknown);

    mlk::CapabilityFlags symCaps;
    symCaps.set(mlk::Capability::HasSymbolicValues);
    symCaps.set(mlk::Capability::HasExactArithmetic);
    symCaps.set(mlk::Capability::HasFunctionValues);
    symCaps.set(mlk::Capability::HasAlgebraicRewriting);
    symCaps.set(mlk::Capability::HasProofCertificates);
    writeProfile((std::string(dir) + "/symbolic_real").c_str(),
                 "symbolic_real", symCaps, false, mlk::TriState::True);

    mlk::CapabilityFlags calcCaps = symCaps;
    calcCaps.set(mlk::Capability::HasCalculus);
    calcCaps.set(mlk::Capability::HasDerivatives);
    calcCaps.set(mlk::Capability::HasIntegrals);
    calcCaps.set(mlk::Capability::HasLimits);
    calcCaps.set(mlk::Capability::HasGradients);
    writeProfile((std::string(dir) + "/calculus_real").c_str(),
                 "calculus_real", calcCaps, false, mlk::TriState::True);

    return 0;
}
