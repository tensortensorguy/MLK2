// MLK+ backend artifact driver (docs/kernel_abi.md #backend-abi;
// docs/polyhedral_spec.md #backend). Closes the AOT loop: emit
// (C++ or x86-64 assembly) -> compile out-of-process with the system
// toolchain -> load with dlopen -> invoke through the buffer ABI.
//
// Publication model (ADR-0003 as amended by ADR-0006): there is NO
// in-process machine codegen anywhere — no emitter-owned W^X pages, no
// patching (Rule 118 is vacuously satisfied), no JIT-spraying surface
// (Rule 123). The artifact is a plain file produced by the external
// toolchain and mapped file-backed by the system loader, exactly like
// any plugin. The polyhedral layer owns the schedule; the artifact is
// the compiled form of the SAME contract the buffer executor walks, so
// the differential test against the walker is bit-exact by construction.
//
// ABI (mirrors the emitter headers):
//   1. Multi-dim/polyhedral modules —
//        extern "C" int mlk_kernel(
//            const double* A, const int64_t* A_dims,   // per bindable
//            ...                                        // buffer, table
//                                                       // order
//            const double* mlk_scalars,                 // FIXED ABI:
//            int64_t mlk_n_scalars)                     // always last
//      returning 0 on success, nonzero on a structural violation
//      (currently 1 = negative store address; the walker reports the
//      same condition as InvalidGraph — the driver maps any nonzero
//      back to an error Result).
//   2. Legacy 1-D modules —
//        extern "C" void mlk_kernel(...buffers..., int64_t n,
//                                   const double* mlk_scalars,
//                                   int64_t mlk_n_scalars)
//
// In BOTH forms the scalars pair is part of the fixed signature, so the
// dispatch and the artifact can never disagree about the arity (an
// empty table degrades scalar reads to 0.0 exactly like the walker).
//
// Binding rules (exactly the walker's resolveInput/resolveOutput):
// input buffers bind io.inputs[bid]; output buffers bind
// io.outputs[bid - io.inputs.size()]; the dims arrays are the module's
// own logical dims (the executor trusts the binder — kernel_abi.md).
// The dispatch covers up to 8 bindable buffers (+ scalars = 18 flat
// arguments); larger modules are rejected honestly, not mis-invoked.
#pragma once

#include <string>

#include "mlk/backend/cpp_emitter.h"
#include "mlk/core/result.h"
#include "mlk/runtime/execution.h"

namespace mlk {

/// Artifact form to build (Rule 78-style explicit selection; the
/// emitters are independently verified).
enum class ArtifactKind : uint8_t {
    Cpp = 0,  // emitCppSource  -> cc -shared -fPIC -fno-exceptions
    Asm = 1,  // emitAsmSource  -> cc -shared (AT&T x86-64)
};

/// Driver knobs (Rule 27: named, documented; all cold-path).
struct BackendDriverConfig {
    /// Compiler driver used to build the shared object. Must accept
    /// `.s`/`.cpp` inputs like cc/gcc/clang.
    std::string compiler{"cc"};
    /// Base directory for per-build workdirs. Empty resolves to
    /// $MLK_BACKEND_WORKDIR, then "<cwd>/mlk_backend_work" (never /tmp:
    /// build artifacts must not depend on tmpfs noexec policy).
    std::string workdirBase{};
};

/// True when the configured compiler is resolvable on PATH (cold probe;
/// used by tests to SKIP honestly when the platform has no toolchain —
/// every other failure is an error, never a silent skip).
[[nodiscard]] bool toolchainAvailable(const BackendDriverConfig& config);

/// A compiled artifact loaded into this process. Owns the dlopen handle
/// (closed on destruction / reassignment; move-only).
class LoadedKernel {
public:
    LoadedKernel() noexcept = default;
    ~LoadedKernel();
    LoadedKernel(LoadedKernel&& other) noexcept;
    LoadedKernel& operator=(LoadedKernel&& other) noexcept;
    LoadedKernel(const LoadedKernel&) = delete;
    LoadedKernel& operator=(const LoadedKernel&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] bool isMultiDim() const noexcept { return multiDim_; }
    [[nodiscard]] const std::string& artifactPath() const noexcept {
        return artifactPath_;
    }
    [[nodiscard]] const std::string& libraryPath() const noexcept {
        return libraryPath_;
    }

    /// Invokes the kernel over the buffer bindings (walker-equivalent
    /// argument materialization; see the binding rules above).
    [[nodiscard]] Result<void> run(const KernelModule& kernel,
                                   SymbolTable& symbols,
                                   const KernelBufferBindings& io) const;

private:
    friend Result<LoadedKernel> buildKernelArtifact(
        const KernelModule& kernel, SymbolTable& symbols,
        ArtifactKind kind, const BackendDriverConfig& config);
    void* handle_{nullptr};
    void* fn_{nullptr};
    bool multiDim_{false};
    std::string artifactPath_{};
    std::string libraryPath_{};
};

/// Emits the artifact (per kind), compiles it out-of-process in a fresh
/// workdir, and dlopens it. Every failure carries its stage: emit /
/// workdir / write / compiler (with the toolchain log tail) / load /
/// symbol.
[[nodiscard]] Result<LoadedKernel> buildKernelArtifact(
    const KernelModule& kernel, SymbolTable& symbols, ArtifactKind kind,
    const BackendDriverConfig& config);

}  // namespace mlk
