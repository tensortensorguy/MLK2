// mlk-poly bench — polyhedral runtime benchmark (Rule 49 protocol).
//
//   mlk-poly bench [--reps=R] [--warmup=W]
//                  [--suites=gemm,softmax,reducesum]
//                  [--json-out=F] [--workdir=D]
//
// Every suite case runs four execution paths over the SAME seeded
// inputs (Rule 56: structured synthetic data, never all-zero):
//   tier1   the Call baseline executed by the buffer executor
//   walker  the Tier2 polyhedral kernel (synth -> ... -> codegen)
//           executed by the MultiDimWalker
//   cpp     the Tier2 kernel emitted as C++, compiled out-of-process,
//           loaded with dlopen (ADR-0003/0006)
//   asm     the Tier2 kernel emitted as x86-64 assembly, assembled
//           out-of-process, loaded with dlopen
// with a bit-exact gate (path output vs the Tier1 reference, Rule 43
// element semantics: NaN never matches; x != y is the whole check)
// BEFORE any timing, then Rule 49 statistics: median/min/max of reps
// timed runs after warmups, steady_clock.
//
// The FLOP model is the spec's declared essential-work model (docs/
// polyhedral_spec.md §fast-kernel-search): 2*M*K*N (MatMul), 5*M*K
// (softmax row model), M*K (ReduceSum). GFLOP/s derives from it — a
// model, never a hardware claim (Axiom 14.21: no "fastest possible").
//
// Honest boundaries (Rule 30): a path that cannot run (transform
// failure, artifact build failure, missing toolchain) is reported as
// a skipped row with the structured reason and never timed; the
// walker failing is a regression (exit 1), a native artifact failing
// to BUILD on a toolchain-less platform is an honest skip.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mlk/backend/backend_driver.h"
#include "mlk/core/constants.h"
#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/pass/register_all.h"
#include "mlk/poly/workspace.h"
#include "mlk/runtime/execution.h"
#include "mlk/support/kernel_ir.h"

#include "poly_bench.h"

namespace polybench {
namespace {

using Clock = std::chrono::steady_clock;

double nowSeconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch())
        .count();
}

double medianOf(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/// One benchmark row: a (suite case, execution path) pair.
struct Row {
    std::string suite{};
    std::string shape{};
    std::string path{};
    double medianMs{0.0};
    double minMs{0.0};
    double maxMs{0.0};
    double speedup{0.0};   // tier1 median / this median
    double gflops{0.0};    // declared model over this median
    double flops{0.0};     // the case's declared essential-work model
    bool bitexact{false};  // gate vs the tier1 reference
    bool timed{false};     // false => skipped, see note
    std::string note{};    // honest skip/failure reason (Rule 30)
};

/// A suite case: builder + shape label + declared flop model.
struct Case {
    std::string suite{};
    std::string shape{};
    int64_t m{0}, k{0}, n{0};  // n unused by the 2-D suites
    int64_t outElements{0};    // exact output element count (ABI)
    double flops{0.0};         // declared essential-work model
    int inArity{1};            // 2 for MatMul, 1 otherwise
};

/// Builds the baseline Call module for one case (the same construction
/// the unit tests use: one root Call node over rank-2 buffers).
void buildCallModule(mlk::SymbolTable& symbols, const Case& c,
                     mlk::KernelModule& km) {
    mlk::KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = mlk::SmallVector<int64_t, 4>{c.m, c.k};
    a.isInput = true;
    const uint32_t bufA = km.addBuffer(a);
    uint32_t bufB = mlk::constants::kInvalidId;
    if (c.inArity == 2) {
        mlk::KernelBuffer b;
        b.name = symbols.intern("B");
        b.dims = mlk::SmallVector<int64_t, 4>{c.k, c.n};
        b.isInput = true;
        bufB = km.addBuffer(b);
    }
    mlk::KernelBuffer out;
    out.name = symbols.intern(c.inArity == 2 ? "C" : "Y");
    // Output rank by op: MatMul -> [M,N]; Softmax -> [M,K] (rank-2 in
    // and out); ReduceSum -> [M] (rank-2 in, rank-1 out).
    if (c.inArity == 2) {
        out.dims = mlk::SmallVector<int64_t, 4>{c.m, c.n};
    } else if (c.suite == "softmax") {
        out.dims = mlk::SmallVector<int64_t, 4>{c.m, c.k};
    } else {
        out.dims = mlk::SmallVector<int64_t, 4>{c.m};
    }
    out.isOutput = true;
    const uint32_t bufOut = km.addBuffer(out);
    mlk::KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = c.suite == "gemm"    ? mlk::MathOp::MatMul
                : c.suite == "softmax" ? mlk::MathOp::Softmax
                                       : mlk::MathOp::ReduceSum;
    call.bufferA = bufA;
    call.bufferB = bufB;
    call.bufferOut = bufOut;
    (void)km.addNode(call);
}

/// The Tier2 polyhedral chain (exactly the unit-test pattern): the
/// seven poly passes over the Call module, in place.
bool transformTier2(mlk::KernelModule& km, mlk::SymbolTable& symbols,
                    mlk::DiagnosticEngine& diag, std::string& reason) {
    mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
    if (ws == nullptr) {
        reason = "workspace-alloc-failed";
        return false;
    }
    mlk::PassContext ctx;
    ctx.symbols = &symbols;
    ctx.diag = &diag;
    ctx.tier = mlk::Tier::Tier2;
    ctx.polyWorkspace = ws;
    ctx.kernelOut = &km;
    mlk::MathGraph g2(&symbols);
    bool ok = true;
    for (const char* name :
         {"poly.synth", "poly.scop_detect", "poly.dependence",
          "poly.schedule", "poly.tile", "poly.codegen", "poly.verify"}) {
        mlk::Pass* p =
            mlk::PassRegistry::instance().byName(symbols,
                                                 symbols.intern(name));
        if (p == nullptr) {
            reason = std::string("pass-missing:") + name;
            ok = false;
            break;
        }
        auto r = p->run(ctx, g2);
        if (!r.has_value()) {
            reason = std::string("pass-failed:") + name;
            ok = false;
            break;
        }
    }
    mlk::poly::destroyPolyWorkspace(ws);
    return ok;
}

/// Seeds one input buffer (Rule 56: deterministic, structured).
void seedInput(mlk::SmallVector<double, 8>& buf, const uint64_t salt) {
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<double>((i * salt) % 13) * 0.25 - 1.0;
    }
}

/// Rule 43 element gate: bit-exact means no element differs under !=.
/// On mismatch, reports the first differing index to stderr (bench
/// diagnostics; the verdict itself stays a plain bool).
bool bitExact(const mlk::SmallVector<double, 8>& got,
              const mlk::SmallVector<double, 8>& ref) {
    if (got.size() != ref.size()) return false;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (got[i] != ref[i]) {
            std::fprintf(stderr,
                         "mlk-poly bench: first mismatch at [%zu]: "
                         "got %.17g ref %.17g\n",
                         i, got[i], ref[i]);
            return false;  // NaN never matches
        }
    }
    return true;
}

/// One timed execution of `run` after the gate; returns seconds.
template <typename Fn>
bool timedPath(Fn&& run, const uint32_t warmup, const uint32_t reps,
               Row& row) {
    for (uint32_t w = 0; w < warmup; ++w) {
        if (!run()) return false;
    }
    std::vector<double> times;
    times.reserve(reps);
    for (uint32_t r = 0; r < reps; ++r) {
        const double t0 = nowSeconds();
        if (!run()) return false;
        const double t1 = nowSeconds();
        times.push_back(t1 - t0);
    }
    row.minMs = *std::min_element(times.begin(), times.end()) * 1000.0;
    row.maxMs = *std::max_element(times.begin(), times.end()) * 1000.0;
    row.medianMs = medianOf(times) * 1000.0;
    row.timed = true;
    return true;
}

void printTable(const std::vector<Row>& rows) {
    std::printf("\n%-9s %-13s %-7s %10s %10s %9s %9s %s\n", "suite",
                "shape", "path", "median_ms", "min_ms", "speedup",
                "gflops", "bitexact note");
    for (const Row& r : rows) {
        std::printf("%-9s %-13s %-7s ", r.suite.c_str(), r.shape.c_str(),
                    r.path.c_str());
        if (!r.timed) {
            std::printf("%10s %10s %9s %9s %-7s %s\n", "-", "-", "-", "-",
                        "-", r.note.c_str());
            continue;
        }
        std::printf("%10.4f %10.4f %8.2fx %9.3f %-7s %s\n", r.medianMs,
                    r.minMs, r.speedup, r.gflops,
                    r.bitexact ? "OK" : "FAIL", r.note.c_str());
    }
}

void writeJson(const std::vector<Row>& rows, const std::string& path,
               const uint32_t reps, const uint32_t warmup) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "mlk-poly bench: cannot write %s\n",
                     path.c_str());
        return;
    }
    std::fprintf(f,
                 "{\n  \"protocol\": {\"reps\": %u, \"warmup\": %u},\n"
                 "  \"rows\": [\n",
                 reps, warmup);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        std::fprintf(f,
                     "    {\"suite\": \"%s\", \"shape\": \"%s\", "
                     "\"path\": \"%s\", \"median_ms\": %.6f, "
                     "\"min_ms\": %.6f, \"max_ms\": %.6f, "
                     "\"speedup\": %.4f, \"gflops\": %.4f, "
                     "\"bitexact\": %s, \"timed\": %s, \"note\": \"%s\"}%s\n",
                     r.suite.c_str(), r.shape.c_str(), r.path.c_str(),
                     r.medianMs, r.minMs, r.maxMs, r.speedup, r.gflops,
                     r.bitexact ? "true" : "false",
                     r.timed ? "true" : "false", r.note.c_str(),
                     i + 1 < rows.size() ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("mlk-poly bench: json report written to %s\n",
                path.c_str());
}

int runSuiteCase(const Case& c, mlk::SymbolTable& symbols,
                 mlk::DiagnosticEngine& diag, const uint32_t warmup,
                 const uint32_t reps, const std::string& workdir,
                 std::vector<Row>& rows) {
    const std::size_t nA = static_cast<std::size_t>(c.m * c.k);
    const std::size_t nB = static_cast<std::size_t>(c.k * c.n);
    // Output element count by op ABI: MatMul [M,N]; Softmax [M,K];
    // ReduceSum [M]. Sizing this wrong under-allocates every path's
    // output buffer (heap corruption — found by the softmax suite).
    const std::size_t nOut = static_cast<std::size_t>(c.outElements);

    // Shared seeded inputs (Rule 56); every path binds the same data.
    mlk::SmallVector<double, 8> inA(nA), inB(nB);
    seedInput(inA, 7);
    seedInput(inB, 5);

    // --- tier1 baseline: reference outputs + timing -------------------
    Row tier1;
    tier1.suite = c.suite;
    tier1.shape = c.shape;
    tier1.path = "tier1";
    tier1.flops = c.flops;
    mlk::SmallVector<double, 8> refOut(nOut, 0.0);
    {
        mlk::KernelModule base;
        buildCallModule(symbols, c, base);
        const bool needB = c.inArity == 2;
        auto bind = [&](mlk::SmallVector<double, 8>& outStorage) {
            // The tier1 Call(MatMul) path accumulates into C (beta=1
            // primitive); the caller-side per-call re-zero IS the beta=0
            // usage the other paths perform via their init statements.
            // Without it, timing reps would compound (the walker row is
            // then wrongly compared against a multiplied reference).
            if (c.inArity == 2) {
                std::fill(outStorage.begin(), outStorage.end(), 0.0);
            }
            mlk::KernelBufferBindings io;
            io.inputs.push_back(inA.data());
            if (needB) io.inputs.push_back(inB.data());
            io.outputs.push_back(outStorage.data());
            io.elements = static_cast<int64_t>(nA);
            return io;
        };
        auto io = bind(refOut);
        auto r = mlk::executeKernelOnBuffers(base, symbols, io, nullptr);
        if (!r.has_value()) {
            tier1.note = "tier1-exec-failed";
            rows.push_back(tier1);
            std::fprintf(stderr,
                         "mlk-poly bench: %s %s tier1 failed: %s\n",
                         c.suite.c_str(), c.shape.c_str(),
                         r.error().message.c_str());
            return 1;
        }
        auto run = [&]() {
            auto io2 = bind(refOut);
            return mlk::executeKernelOnBuffers(base, symbols, io2, nullptr)
                .has_value();
        };
        if (!timedPath(run, warmup, reps, tier1)) {
            tier1.note = "tier1-exec-failed";
            tier1.timed = false;
            rows.push_back(tier1);
            return 1;
        }
        tier1.bitexact = true;  // this row IS the reference
        tier1.note = "reference";
    }
    rows.push_back(tier1);

    // --- Tier2 transform (one shared transformed module) ---------------
    mlk::KernelModule poly;
    buildCallModule(symbols, c, poly);
    std::string reason;
    if (!transformTier2(poly, symbols, diag, reason)) {
        // Honest row (Rule 30): walker is a regression -> exit 1.
        Row w;
        w.suite = c.suite;
        w.shape = c.shape;
        w.path = "walker";
        w.flops = c.flops;
        w.note = reason;
        rows.push_back(w);
        return 1;
    }

    // --- walker --------------------------------------------------------
    Row walk;
    walk.suite = c.suite;
    walk.shape = c.shape;
    walk.path = "walker";
    walk.flops = c.flops;
    {
        mlk::SmallVector<double, 8> outW(nOut, 0.0);
        auto bind = [&]() {
            mlk::KernelBufferBindings io;
            io.inputs.push_back(inA.data());
            if (c.inArity == 2) io.inputs.push_back(inB.data());
            io.outputs.push_back(outW.data());
            io.elements = static_cast<int64_t>(nA);
            return io;
        };
        auto r = mlk::executeKernelOnBuffers(poly, symbols, bind(), nullptr);
        if (!r.has_value()) {
            walk.note = "walker-exec-failed";
            rows.push_back(walk);
            return 1;
        }
        walk.bitexact = bitExact(outW, refOut);
        if (!walk.bitexact) {
            walk.note = "bitexact-FAIL";
            rows.push_back(walk);
            return 1;
        }
        auto run = [&]() {
            return mlk::executeKernelOnBuffers(poly, symbols, bind(),
                                               nullptr)
                .has_value();
        };
        if (!timedPath(run, warmup, reps, walk)) {
            walk.note = "walker-exec-failed";
            walk.timed = false;
            rows.push_back(walk);
            return 1;
        }
    }
    rows.push_back(walk);

    // --- native artifacts (cpp + asm) ----------------------------------
    mlk::BackendDriverConfig cfg;
    if (!workdir.empty()) cfg.workdirBase = workdir;
    if (!toolchainAvailable(cfg)) {
        for (const char* p : {"cpp", "asm"}) {
            Row r;
            r.suite = c.suite;
            r.shape = c.shape;
            r.path = p;
            r.flops = c.flops;
            r.note = "toolchain-unavailable";
            rows.push_back(r);
        }
        return 0;
    }
    const mlk::ArtifactKind kinds[2] = {mlk::ArtifactKind::Cpp,
                                        mlk::ArtifactKind::Asm};
    const char* names[2] = {"cpp", "asm"};
    for (int i = 0; i < 2; ++i) {
        Row nat;
        nat.suite = c.suite;
        nat.shape = c.shape;
        nat.path = names[i];
        nat.flops = c.flops;
        auto loaded = mlk::buildKernelArtifact(poly, symbols, kinds[i], cfg);
        if (!loaded.has_value()) {
            nat.note = "artifact-failed";
            rows.push_back(nat);
            continue;
        }
        mlk::SmallVector<double, 8> outN(nOut, 0.0);
        auto bind = [&]() {
            mlk::KernelBufferBindings io;
            io.inputs.push_back(inA.data());
            if (c.inArity == 2) io.inputs.push_back(inB.data());
            io.outputs.push_back(outN.data());
            io.elements = static_cast<int64_t>(nA);
            return io;
        };
        auto r0 = loaded->run(poly, symbols, bind());
        if (!r0.has_value()) {
            nat.note = "native-run-failed";
            rows.push_back(nat);
            continue;
        }
        if (!bitExact(outN, refOut)) {
            nat.note = "bitexact-FAIL";
            nat.bitexact = false;
            rows.push_back(nat);
            continue;
        }
        nat.bitexact = true;
        auto run = [&]() {
            return loaded->run(poly, symbols, bind()).has_value();
        };
        if (!timedPath(run, warmup, reps, nat)) {
            nat.note = "native-run-failed";
            nat.timed = false;
            rows.push_back(nat);
            continue;
        }
        rows.push_back(nat);
    }
    return 0;
}

std::vector<Case> defaultCases() {
    std::vector<Case> cases;
    // gemm: 2*M*K*N flops (declared model).
    for (const auto& s :
         std::vector<std::array<int64_t, 3>>{{64, 48, 56},
                                             {128, 128, 128},
                                             {256, 256, 256}}) {
        Case c;
        c.suite = "gemm";
        c.shape = std::to_string(s[0]) + "x" + std::to_string(s[1]) + "x" +
                  std::to_string(s[2]);
        c.m = s[0];
        c.k = s[1];
        c.n = s[2];
        c.outElements = s[0] * s[2];
        c.flops = 2.0 * static_cast<double>(s[0] * s[1] * s[2]);
        c.inArity = 2;
        cases.push_back(c);
    }
    // softmax: 5*M*K (max/sub/exp/add/div row model).
    for (const auto& s : std::vector<std::array<int64_t, 2>>{{64, 64},
                                                             {256, 128}}) {
        Case c;
        c.suite = "softmax";
        c.shape = std::to_string(s[0]) + "x" + std::to_string(s[1]);
        c.m = s[0];
        c.k = s[1];
        c.outElements = s[0] * s[1];
        c.flops = 5.0 * static_cast<double>(s[0] * s[1]);
        c.inArity = 1;
        cases.push_back(c);
    }
    // reducesum: M*K adds.
    for (const auto& s : std::vector<std::array<int64_t, 2>>{{256, 256},
                                                             {1024, 64}}) {
        Case c;
        c.suite = "reducesum";
        c.shape = std::to_string(s[0]) + "x" + std::to_string(s[1]);
        c.m = s[0];
        c.k = s[1];
        c.outElements = s[0];
        c.flops = static_cast<double>(s[0] * s[1]);
        c.inArity = 1;
        cases.push_back(c);
    }
    return cases;
}

}  // namespace

int runBench(int argc, char** argv) {
    uint32_t reps = mlk::constants::kBenchDefaultReps;
    uint32_t warmup = mlk::constants::kBenchDefaultWarmup;
    std::string suites = "gemm,softmax,reducesum";
    std::string jsonOut, workdir;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&a](const char* key) -> std::string {
            const std::string prefix = std::string("--") + key + "=";
            return a.rfind(prefix, 0) == 0 ? a.substr(prefix.size())
                                           : std::string();
        };
        if (!val("reps").empty()) {
            reps = static_cast<uint32_t>(std::atoll(val("reps").c_str()));
        } else if (!val("warmup").empty()) {
            warmup =
                static_cast<uint32_t>(std::atoll(val("warmup").c_str()));
        } else if (!val("suites").empty()) {
            suites = val("suites");
        } else if (!val("json-out").empty()) {
            jsonOut = val("json-out");
        } else if (!val("workdir").empty()) {
            workdir = val("workdir");
        } else {
            std::fprintf(stderr, "mlk-poly bench: unknown option %s\n",
                         argv[i]);
            return 2;
        }
    }
    if (reps == 0 || warmup == 0) {
        std::fprintf(stderr, "mlk-poly bench: --reps/--warmup must be > 0\n");
        return 2;
    }
    const bool wantGemm = suites.find("gemm") != std::string::npos;
    const bool wantSoftmax = suites.find("softmax") != std::string::npos;
    const bool wantReduce = suites.find("reducesum") != std::string::npos;
    if (!wantGemm && !wantSoftmax && !wantReduce) {
        std::fprintf(stderr,
                     "mlk-poly bench: --suites must name at least one of "
                     "gemm,softmax,reducesum\n");
        return 2;
    }

    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::DiagnosticEngine diag;

    std::printf("== mlk-poly bench (reps=%u warmup=%u) ==\n", reps, warmup);
    std::vector<Row> rows;
    int exitCode = 0;
    for (const Case& c : defaultCases()) {
        if (c.suite == "gemm" && !wantGemm) continue;
        if (c.suite == "softmax" && !wantSoftmax) continue;
        if (c.suite == "reducesum" && !wantReduce) continue;
        if (runSuiteCase(c, symbols, diag, warmup, reps, workdir, rows) !=
            0) {
            exitCode = 1;
        }
    }
    // Speedup + model throughput derive from the tier1 median.
    for (Row& r : rows) {
        if (!r.timed) continue;
        const Row* base = nullptr;
        for (const Row& q : rows) {
            if (q.timed && q.suite == r.suite && q.shape == r.shape &&
                q.path == "tier1") {
                base = &q;
                break;
            }
        }
        if (base != nullptr && base->medianMs > 0.0 && r.medianMs > 0.0) {
            r.speedup = base->medianMs / r.medianMs;
            // flops / (ms * 1e6) == GFLOP/s.
            r.gflops = r.flops / (r.medianMs * 1.0e6);
        }
    }
    printTable(rows);
    if (!jsonOut.empty()) writeJson(rows, jsonOut, reps, warmup);
    return exitCode;
}

}  // namespace polybench
