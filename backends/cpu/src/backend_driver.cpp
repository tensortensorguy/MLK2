// Backend artifact driver implementation (see header for the ABI and
// publication-model contracts). POSIX-only by design (dlopen/posix_spawn/
// mkdtemp); other platforms fail with UnsupportedCapability at every
// entry point instead of mis-behaving quietly.
#include "mlk/backend/backend_driver.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "mlk/backend/asm_emitter.h"
#include "mlk/core/constants.h"

#if !defined(_WIN32)
// POSIX environment (declared at global scope: a function-local extern
// inside mlk::(anonymous) would bind to the wrong namespace).
extern "C" char** environ;
#endif

namespace mlk {

namespace {

/// Driver dispatch bound: 8 bindable buffers (+ the fixed scalars pair)
/// keep the flat-argument invocation inside a small explicit switch (no
/// libffi, no variadic tricks — every call goes through an exactly
/// typed function pointer).
inline constexpr std::size_t kDriverMaxBindableBuffers = 8;

[[nodiscard]] std::string joinPath(const std::string& a,
                                   const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

[[nodiscard]] Result<std::string> readFileTail(const std::string& path,
                                               const std::size_t tail) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return err(ErrorCode::IoError,
                   "driver: cannot open compiler log: " + path);
    }
    std::string data;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        data.append(buf, n);
        if (data.size() > (1u << 20)) break;  // log sanity cap
    }
    std::fclose(f);
    if (data.size() > tail) {
        data = data.substr(data.size() - tail);
    }
    return data;
}

[[nodiscard]] bool fileExists(const std::string& path) {
#if !defined(_WIN32)
    return ::access(path.c_str(), X_OK) == 0;
#else
    (void)path;
    return false;
#endif
}

/// Writes the artifact text; fopen/fwrite (no exceptions anywhere).
[[nodiscard]] Result<void> writeFile(const std::string& path,
                                     const std::string& content) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return err(ErrorCode::IoError,
                   "driver: cannot write artifact: " + path);
    }
    const std::size_t wrote =
        std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
    if (wrote != content.size()) {
        return err(ErrorCode::IoError,
                   "driver: short write for artifact: " + path);
    }
    return {};
}

#if !defined(_WIN32)

/// Exactly-typed invocation helpers (no variadic tricks; the switch in
/// LoadedKernel::run selects the arity that matches the artifact's flat
/// argument count, scalars pair included).
template <typename... Args>
int callMultiDim(void* fn, Args... args) {
    using F = int (*)(Args...);
    return reinterpret_cast<F>(fn)(args...);
}

template <typename... Args>
void callLegacy(void* fn, Args... args) {
    using F = void (*)(Args...);
    reinterpret_cast<F>(fn)(args...);
}

/// Fresh 0700 workdir under the configured base (never /tmp — build
/// artifacts must live on the project filesystem; see the header).
[[nodiscard]] Result<std::string> makeWorkdir(
    const BackendDriverConfig& config) {
    std::string base = config.workdirBase;
    if (base.empty()) {
        const char* env = std::getenv("MLK_BACKEND_WORKDIR");
        if (env != nullptr && env[0] != '\0') {
            base = env;
        } else {
            char cwd[4096];
            if (::getcwd(cwd, sizeof(cwd)) == nullptr) {
                return err(ErrorCode::IoError,
                           "driver: cannot resolve cwd for workdir");
            }
            base = joinPath(cwd, "mlk_backend_work");
        }
    }
    if (::mkdir(base.c_str(), 0755) != 0 && errno != EEXIST) {
        return err(ErrorCode::IoError,
                   "driver: cannot create workdir base: " + base);
    }
    std::string templ = joinPath(base, "mlkart-XXXXXX");
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    if (::mkdtemp(buf.data()) == nullptr) {
        return err(ErrorCode::IoError,
                   "driver: mkdtemp failed under " + base);
    }
    return std::string(buf.data());
}

/// Runs the compiler over the artifact; stdout/stderr land in
/// <workdir>/build_log.txt (the error message carries its tail on
/// failure — Rule 67: never opaque).
[[nodiscard]] Result<std::string> runCompiler(
    const BackendDriverConfig& config, const std::string& workdir,
    const std::string& artifactPath, const std::string& libraryPath,
    const bool isAsm) {
    const std::string logPath = joinPath(workdir, "build_log.txt");
    std::vector<std::string> argStorage;
    argStorage.push_back(config.compiler);
    if (isAsm) {
        // Assembly artifacts: assemble + link. The emitted text is
        // already position-independent (RIP-relative rodata, PLT
        // calls), so no codegen flags apply. -lm covers the libm PLT
        // calls (transcendentals); harmless when unreferenced.
        argStorage.push_back("-shared");
        argStorage.push_back(artifactPath);
    } else {
        // C++ artifacts: the SAME no-exception/no-rtti contract as the
        // host build (proves the artifact honors the repo rules) and
        // baseline codegen (no FMA contraction, matching the walker).
        argStorage.push_back("-O2");
        argStorage.push_back("-shared");
        argStorage.push_back("-fPIC");
        argStorage.push_back("-fno-exceptions");
        argStorage.push_back("-fno-rtti");
        argStorage.push_back(artifactPath);
    }
    argStorage.push_back("-o");
    argStorage.push_back(libraryPath);
    argStorage.push_back("-lm");

    std::vector<char*> argv;
    argv.reserve(argStorage.size() + 1);
    for (std::string& a : argStorage) argv.push_back(a.data());
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        return err(ErrorCode::Internal, "driver: spawn actions init");
    }
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO,
                                     logPath.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO,
                                     STDERR_FILENO);
    pid_t pid = -1;
    const int rc =
        posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(),
                     environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) {
        return err(ErrorCode::UnsupportedCapability,
                   "driver: cannot spawn compiler '" + config.compiler +
                       "' (rc=" + std::to_string(rc) + ")");
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        return err(ErrorCode::Internal, "driver: waitpid failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        auto log = readFileTail(logPath, 800);
        return err(
            ErrorCode::InvalidArtifact,
            "driver: compiler failed (" +
                std::string(WIFEXITED(status)
                                ? "exit " +
                                      std::to_string(WEXITSTATUS(status))
                                : "signal") +
                "): " + (log.has_value() ? *log : "<no log>"));
    }
    return logPath;
}

struct LibHandle {
    void* handle{nullptr};
    void* fn{nullptr};
};

/// dlopen + dlsym with stage-labeled errors (single open; the handle is
/// owned by the returned LibHandle).
[[nodiscard]] Result<LibHandle> loadLibrary(
    const std::string& libraryPath) {
    void* handle = ::dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* e = ::dlerror();
        return err(ErrorCode::InvalidArtifact,
                   std::string("driver: dlopen failed: ") +
                       (e != nullptr ? e : "<unknown>"));
    }
    void* fn = ::dlsym(handle, "mlk_kernel");
    if (fn == nullptr) {
        const char* e = ::dlerror();
        ::dlclose(handle);
        return err(ErrorCode::InvalidArtifact,
                   std::string("driver: mlk_kernel symbol missing: ") +
                       (e != nullptr ? e : "<unknown>"));
    }
    return LibHandle{handle, fn};
}

#endif  // !_WIN32

}  // namespace

bool toolchainAvailable(const BackendDriverConfig& config) {
#if !defined(_WIN32)
    if (config.compiler.find('/') != std::string::npos) {
        return fileExists(config.compiler);
    }
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr) return false;
    const std::string path(pathEnv);
    std::size_t pos = 0;
    while (pos <= path.size()) {
        std::size_t next = path.find(':', pos);
        if (next == std::string::npos) next = path.size();
        const std::string dir = path.substr(pos, next - pos);
        if (!dir.empty() && fileExists(joinPath(dir, config.compiler))) {
            return true;
        }
        pos = next + 1;
    }
    return false;
#else
    (void)config;
    return false;
#endif
}

LoadedKernel::~LoadedKernel() {
#if !defined(_WIN32)
    if (handle_ != nullptr) {
        ::dlclose(handle_);  // artifact code allocates nothing; no leaks
    }
#endif
}

LoadedKernel::LoadedKernel(LoadedKernel&& other) noexcept
    : handle_(other.handle_),
      fn_(other.fn_),
      multiDim_(other.multiDim_),
      artifactPath_(std::move(other.artifactPath_)),
      libraryPath_(std::move(other.libraryPath_)) {
    other.handle_ = nullptr;
    other.fn_ = nullptr;
}

LoadedKernel& LoadedKernel::operator=(LoadedKernel&& other) noexcept {
    if (this != &other) {
#if !defined(_WIN32)
        if (handle_ != nullptr) ::dlclose(handle_);
#endif
        handle_ = other.handle_;
        fn_ = other.fn_;
        multiDim_ = other.multiDim_;
        artifactPath_ = std::move(other.artifactPath_);
        libraryPath_ = std::move(other.libraryPath_);
        other.handle_ = nullptr;
        other.fn_ = nullptr;
    }
    return *this;
}

Result<LoadedKernel> buildKernelArtifact(const KernelModule& kernel,
                                         SymbolTable& symbols,
                                         const ArtifactKind kind,
                                         const BackendDriverConfig& config) {
#if !defined(_WIN32)
    // Stage 1: emit.
    Result<std::string> source = kind == ArtifactKind::Asm
                                     ? emitAsmSource(kernel, symbols)
                                     : emitCppSource(kernel, symbols);
    if (!source.has_value()) {
        return std::unexpected<Error>(source.error());
    }
    // Stage 2: workdir.
    auto workdir = makeWorkdir(config);
    if (!workdir.has_value()) {
        return std::unexpected<Error>(workdir.error());
    }
    // Stage 3: write.
    const bool isAsm = kind == ArtifactKind::Asm;
    const std::string artifactPath =
        joinPath(*workdir, isAsm ? "kernel.s" : "kernel.cpp");
    auto wrote = writeFile(artifactPath, *source);
    if (!wrote.has_value()) {
        return std::unexpected<Error>(wrote.error());
    }
    // Stage 4: out-of-process toolchain.
    const std::string libraryPath = joinPath(*workdir, "libmlk_kernel.so");
    auto built =
        runCompiler(config, *workdir, artifactPath, libraryPath, isAsm);
    if (!built.has_value()) {
        return std::unexpected<Error>(built.error());
    }
    // Stage 5: load.
    auto lib = loadLibrary(libraryPath);
    if (!lib.has_value()) {
        return std::unexpected<Error>(lib.error());
    }
    LoadedKernel loaded;
    loaded.handle_ = lib->handle;
    loaded.fn_ = lib->fn;
    loaded.multiDim_ = isMultiDimModule(kernel);
    loaded.artifactPath_ = artifactPath;
    loaded.libraryPath_ = libraryPath;
    return loaded;
#else
    (void)kernel;
    (void)symbols;
    (void)kind;
    (void)config;
    return err(ErrorCode::UnsupportedCapability,
               "driver: no POSIX toolchain on this platform");
#endif
}

Result<void> LoadedKernel::run(const KernelModule& kernel,
                               SymbolTable& symbols,
                               const KernelBufferBindings& io) const {
#if !defined(_WIN32)
    if (fn_ == nullptr || handle_ == nullptr) {
        return err(ErrorCode::InvalidArgument,
                   "driver: run on an unloaded kernel");
    }
    (void)symbols;
    // Bindable-buffer pointer materialization — exactly the walker's
    // resolveInput/resolveOutput: inputs index io.inputs by TABLE id,
    // outputs index io.outputs by (bid - nInputs).
    struct PtrPair {
        const double* ptr;
        const int64_t* dims;
    };
    std::vector<PtrPair> pairs;
    std::vector<std::vector<int64_t>> dimsStorage;
    pairs.reserve(kernel.buffers.size());
    dimsStorage.reserve(kernel.buffers.size());
    for (uint32_t bid = 0; bid < kernel.buffers.size(); ++bid) {
        const KernelBuffer& b = kernel.buffers[bid];
        if (!b.isInput && !b.isOutput) continue;
        const double* ptr;
        if (b.isInput) {
            ptr = bid < io.inputs.size() ? io.inputs[bid] : nullptr;
        } else {
            const uint32_t outIdx =
                bid - static_cast<uint32_t>(io.inputs.size());
            ptr = outIdx < io.outputs.size() ? io.outputs[outIdx]
                                             : nullptr;
        }
        if (ptr == nullptr) {
            return err(ErrorCode::InvalidArgument,
                       "driver: buffer not bound by the caller (bid " +
                           std::to_string(bid) + ")");
        }
        dimsStorage.emplace_back(b.dims.begin(), b.dims.end());
        pairs.push_back(PtrPair{ptr, dimsStorage.back().data()});
    }
    if (pairs.size() > kDriverMaxBindableBuffers) {
        return err(ErrorCode::UnsupportedCapability,
                   "driver: module exceeds the " +
                       std::to_string(kDriverMaxBindableBuffers) +
                       "-buffer dispatch bound");
    }
    // The ABI ALWAYS carries the scalars pair (fixed shape — the
    // dispatch and the artifact can never disagree about the arity);
    // unused scalars ride as a zero-length table (reads degrade to 0.0
    // exactly like the walker's empty scalar vector).
    static const int64_t kZeroDim = 0;
    for (PtrPair& p : pairs) {
        if (p.dims == nullptr) p.dims = &kZeroDim;
    }
    const double* scalars =
        io.scalars.empty() ? nullptr : io.scalars.data();
    const int64_t nScalars = static_cast<int64_t>(io.scalars.size());
    static const double kNoScalar = 0.0;
    if (scalars == nullptr) scalars = &kNoScalar;

    if (multiDim_) {
        // int mlk_kernel(ptr0, dims0, ..., scalars, n_scalars)
        const int code = [&]() -> int {
            switch (pairs.size()) {
                case 1:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        scalars, nScalars);
                case 2:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        scalars, nScalars);
                case 3:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        pairs[2].ptr, pairs[2].dims,
                                        scalars, nScalars);
                case 4:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        pairs[2].ptr, pairs[2].dims,
                                        pairs[3].ptr, pairs[3].dims,
                                        scalars, nScalars);
                case 5:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        pairs[2].ptr, pairs[2].dims,
                                        pairs[3].ptr, pairs[3].dims,
                                        pairs[4].ptr, pairs[4].dims,
                                        scalars, nScalars);
                case 6:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        pairs[2].ptr, pairs[2].dims,
                                        pairs[3].ptr, pairs[3].dims,
                                        pairs[4].ptr, pairs[4].dims,
                                        pairs[5].ptr, pairs[5].dims,
                                        scalars, nScalars);
                case 7:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        pairs[2].ptr, pairs[2].dims,
                                        pairs[3].ptr, pairs[3].dims,
                                        pairs[4].ptr, pairs[4].dims,
                                        pairs[5].ptr, pairs[5].dims,
                                        pairs[6].ptr, pairs[6].dims,
                                        scalars, nScalars);
                case 8:
                    return callMultiDim(fn_, pairs[0].ptr, pairs[0].dims,
                                        pairs[1].ptr, pairs[1].dims,
                                        pairs[2].ptr, pairs[2].dims,
                                        pairs[3].ptr, pairs[3].dims,
                                        pairs[4].ptr, pairs[4].dims,
                                        pairs[5].ptr, pairs[5].dims,
                                        pairs[6].ptr, pairs[6].dims,
                                        pairs[7].ptr, pairs[7].dims,
                                        scalars, nScalars);
                default:
                    return -1;
            }
        }();
        if (code != 0) {
            return err(ErrorCode::InvalidGraph,
                       "artifact reported a structural violation "
                       "(code " + std::to_string(code) + ")");
        }
        return {};
    }

    // void mlk_kernel(ptr0, ..., n, scalars, n_scalars) — the legacy
    // elements count rides on io.elements.
    switch (pairs.size()) {
        case 1:
            callLegacy(fn_, pairs[0].ptr, io.elements, scalars, nScalars);
            return {};
        case 2:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, io.elements,
                       scalars, nScalars);
            return {};
        case 3:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, pairs[2].ptr,
                       io.elements, scalars, nScalars);
            return {};
        case 4:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, pairs[2].ptr,
                       pairs[3].ptr, io.elements, scalars, nScalars);
            return {};
        case 5:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, pairs[2].ptr,
                       pairs[3].ptr, pairs[4].ptr, io.elements, scalars,
                       nScalars);
            return {};
        case 6:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, pairs[2].ptr,
                       pairs[3].ptr, pairs[4].ptr, pairs[5].ptr,
                       io.elements, scalars, nScalars);
            return {};
        case 7:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, pairs[2].ptr,
                       pairs[3].ptr, pairs[4].ptr, pairs[5].ptr,
                       pairs[6].ptr, io.elements, scalars, nScalars);
            return {};
        case 8:
            callLegacy(fn_, pairs[0].ptr, pairs[1].ptr, pairs[2].ptr,
                       pairs[3].ptr, pairs[4].ptr, pairs[5].ptr,
                       pairs[6].ptr, pairs[7].ptr, io.elements, scalars,
                       nScalars);
            return {};
        default:
            return err(ErrorCode::UnsupportedCapability,
                       "driver: unhandled buffer count");
    }
#else
    (void)kernel;
    (void)symbols;
    (void)io;
    return err(ErrorCode::UnsupportedCapability,
               "driver: no POSIX toolchain on this platform");
#endif
}

}  // namespace mlk
