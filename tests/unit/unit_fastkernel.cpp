// Fast-kernel search tests (OEIA/Refined/FastKernels, sections
// XIV-XVIII): certificate bundles, budget gates, justification
// decisions, roofline model, launch coverage, artifact-cache validity.
// Every claim asserted here traces to a checkable certificate field —
// the test suite IS the certificate auditor.
#include <chrono>
#include <cstring>
#include <string>

#include "mlk/backend/backend_driver.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/fastkernel/fast_kernel.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"
#include "mlk/runtime/execution.h"
#include "mlk/type/domain_profile.h"

#include "mlk_test.h"

namespace {

/// Cold, cached probe: native-artifact candidates require an
/// out-of-process toolchain; without one they are honestly REJECTED in
/// the search (BuildFailed/CapabilityUnsupported are report outcomes,
/// Axiom 14.20), and cache tests skip (the skip condition is the
/// recorded check).
bool backendToolchainReady() {
    static const bool ready = [] {
        mlk::BackendDriverConfig cfg;
        return mlk::toolchainAvailable(cfg);
    }();
    return ready;
}

/// Cold, cached probe for the Cuda execution path (the GPU driver's
/// recorded check: compiler on PATH; device presence deliberately NOT
/// probed — it is discovered at run time through the artifact's own
/// structured ABI codes).
bool cudaToolchainReady() {
    static const bool ready = [] {
        mlk::GpuBackendDriverConfig cfg;
        return mlk::cudaToolchainAvailable(cfg);
    }();
    return ready;
}

/// Builds an M*K*N MatMul graph (the searched specification s).
mlk::MathGraph buildGemm(mlk::SymbolTable& symbols, const int64_t m,
                         const int64_t k, const int64_t n) {
    mlk::GraphBuilder b(symbols);
    const auto tA = mlk::MathType::tensorValue(mlk::Dtype::F64, {m, k});
    const auto tB = mlk::MathType::tensorValue(mlk::Dtype::F64, {k, n});
    const mlk::ValueId pa = b.placeholder("A", tA);
    const mlk::ValueId pb = b.placeholder("B", tB);
    auto mm = b.op(mlk::MathOp::MatMul, {pa, pb});
    if (mm.has_value()) b.output(*mm);
    return b.graph();
}

mlk::MathDomainProfile tensorProfile(mlk::SymbolTable& symbols) {
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("tensor_f64_cpu");
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    profile.capabilities.set(mlk::Capability::HasTensorDomain);
    profile.capabilities.set(mlk::Capability::HasKernelFusion);
    profile.capabilities.set(mlk::Capability::HasAlgebraicRewriting);
    profile.capabilities.set(mlk::Capability::HasDerivatives);
    profile.capabilities.set(mlk::Capability::HasApproximation);
    return profile;
}

/// A hand-built GEMM module (buffers A[M,K] in, B[K,N] in, C[M,N] out +
/// one Call node) — no pipeline needed for model tests.
mlk::KernelModule handBuiltGemm(const int64_t m, const int64_t k,
                                const int64_t n) {
    mlk::KernelModule mod;
    mlk::KernelBuffer a;
    a.isInput = true;
    a.dims = {m, k};
    mlk::KernelBuffer b;
    b.isInput = true;
    b.dims = {k, n};
    mlk::KernelBuffer c;
    c.isOutput = true;
    c.dims = {m, n};
    (void)mod.addBuffer(a);
    (void)mod.addBuffer(b);
    (void)mod.addBuffer(c);
    mlk::KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::MatMul;
    call.bufferA = 0;
    call.bufferB = 1;
    call.bufferOut = 2;
    (void)mod.addNode(call);
    return mod;
}

/// Environment with realistic magnitudes for search tests (the gap
/// diagnostic U/L - 1 is only meaningful when U > L, i.e. the
/// environment model is physically consistent).
mlk::fastkernel::EnvironmentModel realisticEnv() {
    mlk::fastkernel::EnvironmentModel env;
    env.bandwidthBytesPerSec = 2.5e10;  // 25 GB/s
    env.throughputOpsPerSec = 2.5e9;    // 2.5 GFLOP/s
    env.hardwareThreads = 2;
    env.descriptor = "fixed-realistic-env";
    return env;
}

/// Environment with exact, round numbers for roofline asserts.
mlk::fastkernel::EnvironmentModel fixedEnv() {
    mlk::fastkernel::EnvironmentModel env;
    env.bandwidthBytesPerSec = 1000.0;  // 1000 B/s
    env.throughputOpsPerSec = 100.0;    // 100 op/s
    env.hardwareThreads = 2;
    env.descriptor = "fixed-test-env";
    return env;
}

}  // namespace

// ---------------------------------------------------------------------------
// Launch-coverage certificate (Axioms 14.13/14.14): the slab arithmetic
// must mirror the buffer executor's decomposition exactly.
// ---------------------------------------------------------------------------

MLK_TEST(fastkernel, launch_coverage_partitions_domain) {
    mlk::SymbolTable symbols;
    // The certificate mirrors the executor's decideThreads: the thread
    // count and the spawn-amortization gate read the LOOP'S OWN trip
    // count, so every case builds its loop with the trip it exercises.
    const auto buildLoop = [&symbols](mlk::KernelModule& mod,
                                      const int64_t trip) {
        mlk::KernelNode loop;
        loop.op = mlk::KernelOp::Loop;
        loop.var = symbols.intern("i");
        loop.begin = 0;
        loop.end = trip;
        loop.parallel = true;
        (void)mod.addNode(loop);
        mlk::KernelNode store;
        store.op = mlk::KernelOp::Store;
        store.math = mlk::MathOp::Add;
        store.bufferOut = mlk::constants::kInvalidId;
        (void)mod.addNode(store);
        mod.nodes[0].children.push_back(1);
    };

    // Case 1: threads=3 over a 16384-trip loop: chunk = ceil(16384/3) =
    // 5462; slabs [0,5461] [5462,10923] [10924,16383] — union equals the
    // domain, pairwise disjoint (Axiom 14.14).
    mlk::KernelModule mod;
    buildLoop(mod, 16384);
    *mod.scheduleParams.findOrInsert(symbols.intern("threads"), nullptr) =
        3;
    const auto cov = mlk::fastkernel::certifyLaunchCoverage(mod, symbols,
                                                            16384);
    MLK_CHECK(cov.covered);
    MLK_CHECK_EQ(cov.threads, 3u);
    MLK_CHECK_EQ(cov.slabs.size(), 3u);
    int64_t covered = 0;
    int64_t expectBegin = 0;
    for (const auto& s : cov.slabs) {
        MLK_CHECK_EQ(s.begin, expectBegin);
        if (s.endInclusive >= s.begin) {
            covered += s.endInclusive - s.begin + 1;
        }
        expectBegin = s.endInclusive + 1;
    }
    MLK_CHECK_EQ(covered, 16384);
    MLK_CHECK_EQ(expectBegin, 16384);

    // Case 2: below the spawn-amortization threshold the executor runs
    // one worker regardless of the override (the trip here is 10).
    mlk::KernelModule smallMod;
    buildLoop(smallMod, 10);
    *smallMod.scheduleParams.findOrInsert(symbols.intern("threads"),
                                          nullptr) = 3;
    const auto small = mlk::fastkernel::certifyLaunchCoverage(
        smallMod, symbols, 10);
    MLK_CHECK(small.covered);
    MLK_CHECK_EQ(small.threads, 1u);

    // Case 3: threads above the executor cap is a RESOURCE VIOLATION
    // (Axiom 14.16): the certificate reports it, never hides it.
    mlk::KernelModule overMod;
    buildLoop(overMod, 16384);
    *overMod.scheduleParams.findOrInsert(symbols.intern("threads"),
                                         nullptr) =
        mlk::constants::kKernelExecMaxThreads + 1;
    const auto over = mlk::fastkernel::certifyLaunchCoverage(
        overMod, symbols, 16384);
    MLK_CHECK(!over.covered);
}


MLK_TEST(fastkernel, launch_coverage_sequential_trivial) {
    mlk::SymbolTable symbols;
    mlk::KernelModule mod;  // no parallel loops at all
    mlk::KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = symbols.intern("i");
    (void)mod.addNode(loop);
    const auto cov =
        mlk::fastkernel::certifyLaunchCoverage(mod, symbols, 12345);
    MLK_CHECK(cov.covered);
    MLK_CHECK_EQ(cov.threads, 1u);
    MLK_CHECK_EQ(cov.slabs.size(), 1u);
    MLK_CHECK_EQ(cov.slabs[0].begin, 0);
    MLK_CHECK_EQ(cov.slabs[0].endInclusive, 12344);
}

// ---------------------------------------------------------------------------
// Essential-work model + roofline lower bound (Axioms 14.11/14.12): the
// declared counting rules, verified exactly.
// ---------------------------------------------------------------------------

MLK_TEST(fastkernel, roofline_gemm_model_exact) {
    const mlk::KernelModule mod = handBuiltGemm(4, 3, 2);
    const mlk::fastkernel::EssentialWork w =
        mlk::fastkernel::essentialWorkModel(mod);
    MLK_CHECK(w.opsKnown);
    MLK_CHECK(w.bytesKnown);
    // O_min = 2*M*K*N = 2*4*3*2 = 48 (the closed-form class model).
    MLK_CHECK_EQ(w.minOps, 48);
    // B_min = 8 B * (M*K + K*N + M*N) = 8 * (12 + 6 + 8) = 208.
    MLK_CHECK_EQ(w.minBytes, 208);

    // L^roof = max(B_min/BW, O_min/T) = max(208/1000, 48/100) = 0.48 s.
    const double l =
        mlk::fastkernel::rooflineLowerBound(w, fixedEnv());
    MLK_CHECK_NEAR(l, 0.48, 1e-12);
}

MLK_TEST(fastkernel, roofline_unknown_extents_stay_unknown) {
    mlk::KernelModule mod;
    mlk::KernelBuffer a;
    a.isInput = true;
    a.elements = mlk::constants::kKernelLoopDynamicBound;  // no dims
    mlk::KernelBuffer c;
    c.isOutput = true;
    c.elements = mlk::constants::kKernelLoopDynamicBound;
    (void)mod.addBuffer(a);
    (void)mod.addBuffer(c);
    mlk::KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    (void)mod.addNode(loop);
    mlk::KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.math = mlk::MathOp::Add;
    (void)mod.addNode(store);
    mod.nodes[0].children.push_back(1);

    const mlk::fastkernel::EssentialWork w =
        mlk::fastkernel::essentialWorkModel(mod);
    MLK_CHECK(!w.bytesKnown);
    // Unknown work => NO roofline certificate (tri-state, Rule 22): the
    // bound returns 0 and no claim may be built on it.
    MLK_CHECK_EQ(mlk::fastkernel::rooflineLowerBound(w, fixedEnv()), 0.0);
}

// ---------------------------------------------------------------------------
// Extra-comptime justification decision (Axioms 14.9/15.7): the pure
// decision math, all three certificate forms + the refusal.
// ---------------------------------------------------------------------------

MLK_TEST(fastkernel, justification_forms) {
    using mlk::fastkernel::evaluateExtraComptimeJustification;
    // Absolute runtime improvement: U_new <= U_base - tau.
    const auto abs = evaluateExtraComptimeJustification(0.90, 1.0, 5.0,
                                                        0.05, 1.0, 1);
    MLK_CHECK(abs.justified);
    MLK_CHECK_NEAR(abs.marginSec, 0.10, 1e-12);

    // Amortized payoff: alpha*dM <= N*dR — 2*10 = 20 <= 4*6 = 24.
    const auto amor = evaluateExtraComptimeJustification(0.94, 1.0, 10.0,
                                                         0.5, 2.0, 4);
    MLK_CHECK(amor.justified);
    MLK_CHECK(amor.marginSec > 0.0);

    // Pareto improvement: strict runtime gain only.
    const auto pareto = evaluateExtraComptimeJustification(0.99, 1.0,
                                                           100.0, 0.5,
                                                           1000.0, 1);
    MLK_CHECK(pareto.justified);
    MLK_CHECK_NEAR(pareto.marginSec, 0.01, 1e-12);

    // No certificate: no gain (dR <= 0) => extra comptime refused.
    const auto none = evaluateExtraComptimeJustification(1.0, 1.0, 10.0,
                                                         0.0, 1.0, 100);
    MLK_CHECK(!none.justified);
    const auto worse = evaluateExtraComptimeJustification(1.1, 1.0, 10.0,
                                                          0.0, 1.0, 100);
    MLK_CHECK(!worse.justified);
}

// ---------------------------------------------------------------------------
// Full search: certificates, gates, honest claims (Axioms 14.2-14.21).
// The Tier2 compile of the GEMM SCoP costs seconds on this class of
// hardware, so the searches below keep the declared space small and use
// EXPLICIT budgets (no auto-B0 compiles).
// ---------------------------------------------------------------------------

MLK_TEST(fastkernel, search_same_comptime_certificates) {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::MathGraph graph = buildGemm(symbols, 16, 12, 14);
    mlk::MathDomainProfile profile = tensorProfile(symbols);

    mlk::fastkernel::FastKernelSearchConfig cfg;
    cfg.tileSizes = {0, 8};
    cfg.execPaths = {mlk::fastkernel::ExecPath::Walker};
    cfg.threadCounts = {0};
    cfg.warmupReps = 1;
    cfg.benchReps = 3;
    cfg.budget.mode = mlk::fastkernel::BudgetMode::SameComptime;
    // Explicit, generous budget: every candidate must compile in.
    cfg.budget.baselineComptimeSec = 600.0;
    cfg.measureEnvironment = false;
    cfg.env = realisticEnv();

    auto report = mlk::fastkernel::runFastKernelSearch(graph, symbols,
                                                       profile, cfg);
    MLK_CHECK(report.has_value());
    if (!report.has_value()) return;
    const auto& r = *report;

    MLK_CHECK_EQ(r.candidates.size(), 2u);
    MLK_CHECK(!r.budgetFailure);
    MLK_CHECK(r.winnerIndex >= 0);
    for (const auto& o : r.candidates) {
        MLK_CHECK(o.admissible);
        MLK_CHECK(o.cert.identityCertified);
        MLK_CHECK_EQ(o.cert.identityMaxAbsDiff, 0.0);  // bit-exact (pi)
        MLK_CHECK(o.cert.runtimeUpperBoundSec > 0.0);
        MLK_CHECK(o.cert.comptimeTotalSec > 0.0);
        MLK_CHECK(o.cert.launch.covered);
        MLK_CHECK(!o.usedExtraComptime);
        MLK_CHECK(o.objectiveSec > 0.0);
    }
    // Winner = smallest certified objective; strict-< tie-break keeps
    // declaration order deterministic (Axiom 14.10).
    const auto& w = r.candidates[static_cast<std::size_t>(r.winnerIndex)];
    for (const auto& o : r.candidates) {
        MLK_CHECK(w.objectiveSec <= o.objectiveSec);
    }
    // Roofline from the specification's essential work (Axiom 14.12):
    // GEMM 16*12*14 => O_min = 2*16*12*14 = 5376.
    MLK_CHECK(r.work.opsKnown);
    MLK_CHECK_EQ(r.work.minOps, 2 * 16 * 12 * 14);
    MLK_CHECK(r.rooflineLowerSec > 0.0);
    MLK_CHECK(r.winnerGapOverLower > 0.0);
    // Honest claim (Axiom 14.21): exhaustive finite search over the
    // declared space — never "fastest possible".
    MLK_CHECK(r.claim.find("fastest certified kernel within the declared "
                           "searched space") != std::string::npos);
    // The report is machine-checkable JSON (Meta-Axiom 0.1).
    const std::string text = mlk::json::serializePretty(r.toJson());
    auto back = mlk::json::parse(text);
    MLK_CHECK(back.has_value());
    MLK_CHECK(back->isObject());
}

MLK_TEST(fastkernel, budget_guard_reports_failure) {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::MathGraph graph = buildGemm(symbols, 16, 12, 14);
    mlk::MathDomainProfile profile = tensorProfile(symbols);

    mlk::fastkernel::FastKernelSearchConfig cfg;
    cfg.tileSizes = {0};
    cfg.execPaths = {mlk::fastkernel::ExecPath::Walker};
    cfg.threadCounts = {0};
    cfg.warmupReps = 1;
    cfg.benchReps = 2;
    cfg.budget.mode = mlk::fastkernel::BudgetMode::SameComptime;
    // A 1 ms compile budget: NOTHING compiles this fast — every
    // candidate must be rejected with the budget reason (Axiom 15.6:
    // report budget failure, never silently overrun).
    cfg.budget.baselineComptimeSec = 0.001;
    cfg.measureEnvironment = false;
    cfg.env = fixedEnv();

    auto report = mlk::fastkernel::runFastKernelSearch(graph, symbols,
                                                       profile, cfg);
    MLK_CHECK(report.has_value());
    if (!report.has_value()) return;
    const auto& r = *report;
    MLK_CHECK_EQ(r.candidates.size(), 1u);
    MLK_CHECK(!r.candidates[0].admissible);
    MLK_CHECK_EQ(r.candidates[0].reject,
                 mlk::fastkernel::RejectReason::ComptimeBudgetExceeded);
    MLK_CHECK(r.budgetFailure);
    MLK_CHECK_EQ(r.winnerIndex, -1);
    MLK_CHECK(r.claim.find("budget failure") != std::string::npos);
}

MLK_TEST(fastkernel, extra_comptime_without_payoff_rejected) {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::MathGraph graph = buildGemm(symbols, 16, 12, 14);
    mlk::MathDomainProfile profile = tensorProfile(symbols);

    mlk::fastkernel::FastKernelSearchConfig cfg;
    cfg.tileSizes = {0};
    cfg.execPaths = {mlk::fastkernel::ExecPath::Walker};
    cfg.threadCounts = {0};
    cfg.warmupReps = 1;
    cfg.benchReps = 2;
    cfg.budget.mode = mlk::fastkernel::BudgetMode::BoundedExtra;
    // B0 = 1 ms (nothing fits), dB_max = 600 s (everything fits) — so
    // every candidate USES extra comptime, and the justification gate
    // decides. tau = 600 s is unreachable and the polyhedral GEMM does
    // not beat the Tier1 blocked baseline, so NO certificate form fires:
    // every candidate must be rejected NoJustification (Rule 17.2
    // fallback = no admissible extra-comptime kernel).
    cfg.budget.baselineComptimeSec = 0.001;
    cfg.budget.extraComptimeMaxSec = 600.0;
    cfg.budget.minRuntimeImprovementSec = 600.0;
    cfg.measureEnvironment = false;
    cfg.env = fixedEnv();

    auto report = mlk::fastkernel::runFastKernelSearch(graph, symbols,
                                                       profile, cfg);
    MLK_CHECK(report.has_value());
    if (!report.has_value()) return;
    const auto& r = *report;
    MLK_CHECK_EQ(r.candidates.size(), 1u);
    MLK_CHECK(!r.candidates[0].admissible);
    MLK_CHECK_EQ(r.candidates[0].reject,
                 mlk::fastkernel::RejectReason::NoJustification);
    MLK_CHECK(r.budgetFailure);
}

// ---------------------------------------------------------------------------
// Artifact-cache validity (Axioms 14.22/15.8): fingerprint-keyed reuse,
// M^reuse vs M^first, and environment-change invalidation.
// ---------------------------------------------------------------------------

MLK_TEST(fastkernel, cache_reuse_and_invalidation) {
    if (!backendToolchainReady()) {
        return;  // honest skip: the recorded toolchain probe is the check
    }
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::MathDomainProfile profile = tensorProfile(symbols);

    // Run-unique cache root: a stale entry from a previous binary run
    // would legitimately register as a HIT and break the miss/hit
    // script of THIS run (the cache is doing its job — the test must
    // start clean).
    // Portable, cwd-relative workroot (never /tmp, no machine-specific
    // paths): ctest runs each suite from its build directory, so "fkwork"
    // lands inside the build tree on every machine (review finding).
    const std::string cacheDir =
        std::string("fkwork/cache_test-") +
        std::to_string(
            static_cast<long long>(::std::chrono::steady_clock::now()
                                       .time_since_epoch()
                                       .count()));

    mlk::fastkernel::FastKernelSearchConfig cfg;
    cfg.tileSizes = {0};
    cfg.execPaths = {mlk::fastkernel::ExecPath::NativeAsm};
    cfg.threadCounts = {0};
    cfg.warmupReps = 1;
    cfg.benchReps = 2;
    cfg.budget.mode = mlk::fastkernel::BudgetMode::SameComptime;
    cfg.budget.baselineComptimeSec = 600.0;
    cfg.cacheDir = cacheDir;
    cfg.driver.workdirBase = "fkwork";
    cfg.measureEnvironment = false;
    cfg.env = fixedEnv();

    mlk::MathGraph graph = buildGemm(symbols, 16, 12, 14);
    auto first = mlk::fastkernel::runFastKernelSearch(graph, symbols,
                                                      profile, cfg);
    MLK_CHECK(first.has_value());
    if (!first.has_value()) return;
    MLK_CHECK_EQ(first->cacheMisses, 1u);
    MLK_CHECK_EQ(first->cacheHits, 0u);
    MLK_CHECK(first->winnerIndex >= 0);
    if (first->winnerIndex >= 0) {
        const auto& w = first->candidates[static_cast<std::size_t>(
            first->winnerIndex)];
        MLK_CHECK(w.admissible);
        MLK_CHECK(w.cert.identityCertified);
        MLK_CHECK(!w.cert.fingerprint.empty());
        MLK_CHECK(!w.cert.comptimeFromCache);
        MLK_CHECK(w.cert.comptimeTotalSec > 0.0);
    }

    // Second run, same fingerprint: the artifact is REUSED — M^reuse is
    // the load time alone (Axiom 15.8) and the cache counters move.
    mlk::MathGraph graph2 = buildGemm(symbols, 16, 12, 14);
    auto second = mlk::fastkernel::runFastKernelSearch(graph2, symbols,
                                                       profile, cfg);
    MLK_CHECK(second.has_value());
    if (!second.has_value()) return;
    MLK_CHECK_EQ(second->cacheHits, 1u);
    MLK_CHECK_EQ(second->cacheMisses, 0u);
    if (second->winnerIndex >= 0) {
        const auto& w2 = second->candidates[static_cast<std::size_t>(
            second->winnerIndex)];
        MLK_CHECK(w2.cert.comptimeFromCache);
        MLK_CHECK_EQ(w2.cert.buildSec, 0.0);
        MLK_CHECK(w2.cert.loadSec > 0.0);
    }

    // Environment change (Axiom 14.22): the descriptor is part of the
    // fingerprint, so a different environment INVALIDATES the entry.
    mlk::fastkernel::EnvironmentModel otherEnv = fixedEnv();
    otherEnv.bandwidthBytesPerSec *= 2.0;  // descriptor string changes
    cfg.env = otherEnv;
    mlk::MathGraph graph3 = buildGemm(symbols, 16, 12, 14);
    auto third = mlk::fastkernel::runFastKernelSearch(graph3, symbols,
                                                      profile, cfg);
    MLK_CHECK(third.has_value());
    if (!third.has_value()) return;
    MLK_CHECK_EQ(third->cacheMisses, 1u);
    MLK_CHECK_EQ(third->cacheHits, 0u);
}

MLK_TEST(fastkernel, cuda_candidate_declared_space) {
    // The Cuda execution path is part of the DECLARED variant space
    // (Axiom 14.20): without the nvcc toolchain its candidate ends in
    // a structured CapabilityUnsupported rejection naming the failed
    // probe (recorded, never silent — Rule 148), and the search still
    // completes with an honest no-winner claim. With the toolchain the
    // same candidate must certify identity + runtime like any native
    // path — or, on a machine with a compiler but no DEVICE, fail at
    // run time through the artifact's own structured ABI code (the
    // discovery is recorded as a BuildFailed rejection detail).
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::MathDomainProfile profile = tensorProfile(symbols);

    mlk::fastkernel::FastKernelSearchConfig cfg;
    cfg.tileSizes = {0};
    cfg.execPaths = {mlk::fastkernel::ExecPath::NativeCuda};
    cfg.threadCounts = {0};
    cfg.warmupReps = 1;
    cfg.benchReps = 2;
    cfg.budget.mode = mlk::fastkernel::BudgetMode::SameComptime;
    cfg.budget.baselineComptimeSec = 600.0;
    cfg.driver.workdirBase = "fkwork";
    cfg.gpu.workdirBase = "fkwork";
    cfg.measureEnvironment = false;
    cfg.env = fixedEnv();

    mlk::MathGraph graph = buildGemm(symbols, 16, 12, 14);
    auto r = mlk::fastkernel::runFastKernelSearch(graph, symbols, profile,
                                                  cfg);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    MLK_CHECK_EQ(r->candidates.size(), 1u);
    const auto& o = r->candidates[0];
    MLK_CHECK_EQ(o.config.exec, mlk::fastkernel::ExecPath::NativeCuda);
    if (cudaToolchainReady()) {
        // Compiler present: the artifact builds through the GPU driver.
        // Without a device the run discovers it (ABI code 4) and the
        // rejection detail records the structured failure.
        if (o.admissible) {
            MLK_CHECK(o.cert.identityCertified);
            MLK_CHECK(o.cert.identityMaxAbsDiff == 0.0);
            MLK_CHECK(o.cert.runtimeCertified);
            MLK_CHECK(o.cert.runtimeUpperBoundSec > 0.0);
            MLK_CHECK(!o.cert.fingerprint.empty());
        } else {
            MLK_CHECK(o.reject ==
                      mlk::fastkernel::RejectReason::BuildFailed);
            MLK_CHECK(!o.rejectDetail.empty());
            MLK_CHECK(r->winnerIndex < 0);
        }
    } else {
        // No toolchain: the structured rejection IS the honest skip.
        MLK_CHECK(o.reject ==
                  mlk::fastkernel::RejectReason::CapabilityUnsupported);
        MLK_CHECK(!o.rejectDetail.empty());
        MLK_CHECK(r->winnerIndex < 0);
        MLK_CHECK(r->budgetFailure);
        MLK_CHECK(r->claim.find("budget failure") != std::string::npos);
    }
}

MLK_TEST_MAIN("fastkernel")
