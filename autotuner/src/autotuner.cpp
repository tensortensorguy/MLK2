// Autotuner implementation (Rules 49, 54-59; realization spec §15).
#include "mlk/autotune/searcher.h"

#include <algorithm>
#include <cmath>
#include <chrono>

#include "mlk/cost/cost_model.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/runtime/execution.h"

namespace mlk {

namespace {
double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

/// Seeded deterministic generator producing realistic, non-degenerate data
/// (Rule 56: no all-zero buffers; sin-composed patterns with scale).
double sampleInput(uint64_t i) {
    const double x = static_cast<double>(i % 4096) / 512.0;
    return 0.5 * std::sin(x) + 0.25 * std::cos(2.0 * x) + 0.5;
}
}  // namespace

json::Value SearchSpace::toJson(SymbolTable& symbols) const {
    json::Value doc = json::Object{};
    json::Value arr = json::Array{};
    for (const auto& p : params) {
        json::Value po = json::Object{};
        po.set("name", json::Value{symbols.text(p.name)});
        json::Value dom = json::Array{};
        for (const int64_t d : p.domain) dom.push(json::Value{d});
        po.set("domain", std::move(dom));
        po.set("prior", json::Value{p.prior});
        arr.push(std::move(po));
    }
    doc.set("params", std::move(arr));
    return doc;
}

Result<SearchSpace> SearchSpace::fromJson(const json::Value& doc,
                                          SymbolTable& symbols) {
    SearchSpace sp;
    const json::Value* arr = doc.find("params");
    if (arr == nullptr || !arr->isArray()) {
        return err(ErrorCode::ParseError, "search space missing params", 54);
    }
    for (const auto& po : arr->asArray()) {
        SearchParam p;
        const json::Value* n = po.find("name");
        if (n == nullptr || !n->isString()) {
            return err(ErrorCode::ParseError, "param missing name", 54);
        }
        p.name = symbols.intern(n->asString());
        const json::Value* dom = po.find("domain");
        if (dom == nullptr || !dom->isArray() || dom->asArray().empty()) {
            return err(ErrorCode::ParseError,
                       "param domain must be non-empty array", 54);
        }
        for (const auto& d : dom->asArray()) {
            if (!d.isInt()) {
                return err(ErrorCode::ParseError, "domain entry not int", 54);
            }
            p.domain.push_back(d.asInt());
        }
        const json::Value* prior = po.find("prior");
        if (prior != nullptr && prior->isInt()) p.prior = prior->asInt();
        sp.params.push_back(p);
    }
    return sp;
}

OpenHashMap<SymbolId, int64_t> SearchSpace::realize(
    const SmallVector<int64_t, 8>& config) const {
    OpenHashMap<SymbolId, int64_t> out;
    const std::size_t n = config.size() < params.size() ? config.size()
                                                        : params.size();
    for (std::size_t i = 0; i < n; ++i) {
        int64_t* slot = out.findOrInsert(params[i].name, nullptr, config[i]);
        *slot = config[i];
    }
    return out;
}

Autotuner::Autotuner(SymbolTable& symbols, TelemetrySink& telemetry)
    : symbols_(symbols), telemetry_(telemetry) {}

SearchSpace Autotuner::buildSpace(const MathGraph& graph) const {
    // Declarative space for elementwise/matmul graphs (spec §5.4 knobs;
    // values from constants — Rule 27; each tunable has a documented
    // rationale and is overridable via knobs, Rule 29).
    SearchSpace sp;
    SearchParam vw;
    vw.name = symbols_.intern("vector_width");
    for (const int64_t v : constants::kVectorWidthChoices) vw.domain.push_back(v);
    vw.prior = static_cast<int64_t>(constants::kDefaultSimdWidthF32);
    sp.params.push_back(vw);

    SearchParam unroll;
    unroll.name = symbols_.intern("unroll");
    for (const int64_t v : constants::kUnrollChoices) unroll.domain.push_back(v);
    unroll.prior = 1;
    sp.params.push_back(unroll);

    SearchParam threads;
    threads.name = symbols_.intern("parallel");
    threads.domain.push_back(1);
    threads.domain.push_back(static_cast<int64_t>(
        HardwareInfo::detect().cores));
    threads.prior = 1;
    sp.params.push_back(threads);

    bool hasMatmul = false;
    for (const NodeId nid : graph.topoOrder()) {
        if (graph.node(nid).op == MathOp::MatMul) hasMatmul = true;
    }
    if (hasMatmul) {
        struct TileDef {
            const char* name;
            int64_t prior;
        };
        static constexpr TileDef kTiles[] = {
            {"tile_m", constants::kDefaultTileM},
            {"tile_n", constants::kDefaultTileN},
            {"tile_k", constants::kDefaultTileK}};
        for (const TileDef& t : kTiles) {
            SearchParam tp;
            tp.name = symbols_.intern(t.name);
            for (const int64_t v : constants::kTileChoices) {
                tp.domain.push_back(v);
            }
            tp.prior = t.prior;
            sp.params.push_back(tp);
        }
    }
    return sp;
}

Result<KernelModule> Autotuner::compileCandidate(
    const MathGraph& graph, const TuningContext& ctx,
    const OpenHashMap<SymbolId, int64_t>& params) const {
    // The previous implementation "verified" and "measured" an EMPTY
    // KernelModule{}: executeKernel re-ran the interpreter, so verification
    // compared the interpreter against itself and every measurement was
    // noise around one fixed workload (Rule 58 violation). Candidates are
    // now compiled through the Tier-1 pipeline and executed as real
    // KernelModules.
    MathGraph candidate = graph;  // Rule 13: the pipeline mutates its copy

    const SymbolId vwAttr = symbols_.intern("vector_width");
    const SymbolId tmAttr = symbols_.intern("tile_m");
    const SymbolId tnAttr = symbols_.intern("tile_n");
    const SymbolId tkAttr = symbols_.intern("tile_k");
    const int64_t* vw = params.find(vwAttr);
    const int64_t* tm = params.find(tmAttr);
    const int64_t* tn = params.find(tnAttr);
    const int64_t* tk = params.find(tkAttr);
    for (Node& node : candidate.nodesRef()) {
        if (node.flags.test(NodeFlag::Dead)) continue;
        auto appendI64 = [&](SymbolId name, int64_t value) {
            for (const Attr& a : node.attrs) {
                if (a.name == name) return;  // keep existing schedule attrs
            }
            if (node.attrs.size() >= kMaxAttrsPerNode) return;
            Attr a;
            a.name = name;
            a.value = AttrValue{value};
            node.attrs.push_back(a);
        };
        if (vw != nullptr) appendI64(vwAttr, *vw);
        if (node.op == MathOp::MatMul) {
            if (tm != nullptr) appendI64(tmAttr, *tm);
            if (tn != nullptr) appendI64(tnAttr, *tn);
            if (tk != nullptr) appendI64(tkAttr, *tk);
        }
    }

    KernelModule kernel;
    DiagnosticEngine diag;
    PipelineRunner runner(symbols_, &telemetry_);
    PassContext ctxPass;
    MathDomainProfile defaultProfile;
    defaultProfile.name = symbols_.intern("scalar_f64");
    defaultProfile.capabilities.set(mlk::Capability::HasNumericValues);
    defaultProfile.capabilities.set(mlk::Capability::HasFloatingPoint);
    ctxPass.domainProfile =
        ctx.profile != nullptr ? ctx.profile : &defaultProfile;
    ctxPass.accuracy = ctx.accuracy;  // null is fine for the Tier-1 set
    ctxPass.diag = &diag;
    ctxPass.telemetry = &telemetry_;
    ctxPass.symbols = &symbols_;
    ctxPass.tier = Tier::Tier1;
    ctxPass.kernelOut = &kernel;
    MLK_TRY_VAR(result, runner.run(Tier::Tier1, ctxPass, candidate, &kernel));
    (void)result;
    if (diag.hasErrors()) {
        return err(ErrorCode::VerificationFailed,
                   "candidate compilation produced diagnostics (Rule 58)",
                   58);
    }
    // Executor-consumable params the lowering does not copy: the parallel
    // knob drives decideThreads through the "threads" schedule param.
    const int64_t* parallel = params.find(symbols_.intern("parallel"));
    if (parallel != nullptr && *parallel >= 1) {
        (void)kernel.scheduleParams.findOrInsert(
            symbols_.intern("threads"), nullptr, *parallel);
    }
    return kernel;
}

Result<bool> Autotuner::verifyCandidate(const MathGraph& graph,
                                        const TuningContext& ctx,
                                        const TuningCandidate& cand) const {
    // Rule 58: correctness precedes benchmarking — compile the CANDIDATE
    // config and compare its kernel execution against the Tier 0 oracle on
    // seeded inputs within the contract. Scalar-ABI only today: a kernel
    // with input buffers (tensor graphs) is rejected by executeKernel and
    // surfaces here as a structured error, never as a fabricated win.
    SmallVector<double, 8> inputs;
    for (std::size_t i = 0; i < 8; ++i) inputs.push_back(sampleInput(i + 1));
    MLK_TRY_VAR(oracle, interpretGraph(graph, inputs));
    const OpenHashMap<SymbolId, int64_t> params = space_.realize(cand.config);
    MLK_TRY_VAR(kernel, compileCandidate(graph, ctx, params));
    MLK_TRY_VAR(got,
                executeKernel(kernel, graph, inputs, nullptr, symbols_));
    if (got.outputScalars.size() != oracle.outputScalars.size()) {
        return false;
    }
    const double tol =
        ctx.accuracy != nullptr ? ctx.accuracy->maxRelError : 1e-9;
    for (std::size_t i = 0; i < oracle.outputScalars.size(); ++i) {
        const double a = oracle.outputScalars[i];
        const double b = got.outputScalars[i];
        if (std::fabs(a - b) > tol * (1.0 + std::fabs(a))) return false;
    }
    return true;
}

Result<BenchmarkMeasurement> Autotuner::measureCandidate(
    const MathGraph& graph, const TuningContext& ctx,
    const TuningCandidate& cand) const {
    // Time the CANDIDATE'S KERNEL (previously: the interpreter was timed
    // identically for every config, so the "winner" was pure noise).
    // Compile once outside the timing loop; every rep executes the real
    // KernelModule through the same scalar ABI the engine uses.
    BenchmarkMeasurement m;
    m.reps = ctx.protocol.reps;
    SmallVector<double, 8> inputs;
    for (std::size_t i = 0; i < 8; ++i) inputs.push_back(sampleInput(i + 7));
    const OpenHashMap<SymbolId, int64_t> params = space_.realize(cand.config);
    MLK_TRY_VAR(compiled, compileCandidate(graph, ctx, params));
    SmallVector<double, 64> samples;
    for (uint32_t r = 0; r < ctx.protocol.warmup + ctx.protocol.reps; ++r) {
        const double t0 = nowMs();
        auto run = executeKernel(compiled, graph, inputs, nullptr, symbols_);
        const double t1 = nowMs();
        if (!run.has_value()) {
            return err(ErrorCode::VerificationFailed,
                       "candidate failed during measurement (Rule 58)", 58);
        }
        if (r >= ctx.protocol.warmup) samples.push_back(t1 - t0);
    }
    if (samples.empty()) {
        return err(ErrorCode::VerificationFailed, "no benchmark samples", 49);
    }
    SmallVector<double, 64> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    m.minMs = sorted.front();
    m.medianMs = sorted[sorted.size() / 2];
    double mean = 0.0;
    for (const double s : samples) mean += s;
    mean /= static_cast<double>(samples.size());
    double var = 0.0;
    for (const double s : samples) var += (s - mean) * (s - mean);
    var /= static_cast<double>(samples.size());
    m.stddevMs = std::sqrt(var);
    m.opsPerSecond = mean > 0.0 ? 1000.0 / mean : 0.0;
    return m;
}

Result<CacheEntry> Autotuner::tune(const MathGraph& graph,
                                   const TuningContext& ctx,
                                   RealizationCache& cache) {
    TelemetryEvent attempt;
    attempt.kind = TelemetryEventKind::TuneAttempt;
    telemetry_.record(attempt);

    space_ = buildSpace(graph);

    // Seed candidate: priors (Rule 54: prior-guided start).
    SmallVector<int64_t, 8> seed;
    for (const auto& p : space_.params) seed.push_back(p.prior);

    SmallVector<TuningCandidate, 16> candidates;
    TuningCandidate base;
    base.config = seed;
    // Neighborhood: single-axis moves (tune.local_search MVP).
    candidates.push_back(base);
    for (std::size_t axis = 0; axis < space_.params.size(); ++axis) {
        for (const int64_t v : space_.params[axis].domain) {
            if (v == seed[axis]) continue;
            TuningCandidate c;
            c.config = seed;
            c.config[axis] = v;
            candidates.push_back(c);
            if (candidates.size() >= ctx.maxCandidates) break;
        }
        if (candidates.size() >= ctx.maxCandidates) break;
    }

    // Rule 50/58: verify all candidates BEFORE any timing.
    SmallVector<TuningCandidate, 16> verified;
    for (auto& c : candidates) {
        MLK_TRY_VAR(ok, verifyCandidate(graph, ctx, c));
        c.verified = ok;
        if (ok) verified.push_back(c);
    }
    if (verified.empty()) {
        TelemetryEvent fail;
        fail.kind = TelemetryEventKind::TuneFailure;
        fail.reason = symbols_.intern("all_candidates_failed_verification");
        telemetry_.record(fail);
        return err(ErrorCode::VerificationFailed,
                   "no candidate passed correctness verification (Rule 58)",
                   58);
    }

    // Rule 55: prune by roofline lower bound before measuring.
    const CostEstimate total = BasicCostModel().graphCost(graph, ctx.hardware);
    for (auto& c : verified) {
        c.costLowerBoundNs = rooflineLowerBoundNs(total, ctx.hardware);
    }
    double bestLower = 1e18;
    for (const auto& c : verified) {
        bestLower = std::min(bestLower, c.costLowerBoundNs);
    }
    SmallVector<TuningCandidate, 16> viable;
    for (auto& c : verified) {
        // Keep everything within a factor of the best lower bound.
        if (c.costLowerBoundNs <= bestLower * 2.0) viable.push_back(c);
    }

    // Measure survivors; Rule 59: noise filter via relative stddev.
    TuningCandidate* best = nullptr;
    for (auto& c : viable) {
        MLK_TRY_VAR(m, measureCandidate(graph, ctx, c));
        c.measured = m;
        const bool noisy =
            m.medianMs > 0.0 &&
            (m.stddevMs / m.medianMs) > constants::kBenchNoiseRelativeStddev;
        if (noisy) continue;
        if (best == nullptr || m.medianMs < best->measured->medianMs) {
            best = &c;
        }
    }
    // Rule 59: ties within noise prefer simpler/lower compile cost — the
    // seed (priors) wins ties because it appears first in `viable`.
    if (best == nullptr && !viable.empty()) best = &viable[0];

    CacheEntry entry;
    entry.key.graphHash = graphHash(graph);
    entry.key.accuracyHash =
        ctx.accuracy != nullptr ? ctx.accuracy->hash() : 0;
    entry.key.profileVersion = ctx.profile != nullptr
                                   ? ctx.profile->dialectHash
                                   : 0;
    entry.key.hardwareFingerprint = ctx.hardware.fingerprint();
    entry.key.passPipelineHash = symbols_.intern("tier2_default");
    entry.key.propertyFactsHash = 0;
    entry.key.superoptVersion = 0;
    entry.key.runtimeConfig = 0;
    entry.key.shapeBucket = 0;
    entry.strategy = symbols_.intern("local_search");
    entry.kernelHash = 0;
    entry.schedule = space_.toJson(symbols_);
    entry.measuredMs = best != nullptr && best->measured.has_value()
                           ? best->measured->medianMs
                           : 0.0;
    cache.store(entry);
    return entry;
}

}  // namespace mlk

// --- benchmarkGraph (benchmarker.h contract; Rule 49 protocol) -------------
#include <cstdlib>

namespace mlk {

namespace {
double benchNowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}
}  // namespace

Result<BenchmarkMeasurement> benchmarkGraph(const MathGraph& graph,
                                            SymbolTable& symbols,
                                            const BenchmarkProtocol& p) {
    BenchmarkMeasurement m;
    m.reps = p.reps;
    SmallVector<double, 8> inputs;
    for (std::size_t i = 0; i < 8; ++i) inputs.push_back(sampleInput(i + 3));
    SmallVector<double, 64> samples;
    for (uint32_t r = 0; r < p.warmup + p.reps; ++r) {
        const double t0 = benchNowMs();
        auto run = interpretGraph(graph, inputs);
        if (!run.has_value()) {
            return err(ErrorCode::VerificationFailed,
                       "graph failed during benchmark (Rule 58)", 58);
        }
        const double t1 = benchNowMs();
        if (r >= p.warmup) samples.push_back(t1 - t0);
    }
    if (samples.empty()) {
        return err(ErrorCode::VerificationFailed, "no samples", 49);
    }
    SmallVector<double, 64> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    m.minMs = sorted.front();
    m.medianMs = sorted[sorted.size() / 2];
    double mean = 0.0;
    for (const double s : samples) mean += s;
    mean /= static_cast<double>(samples.size());
    double var = 0.0;
    for (const double s : samples) var += (s - mean) * (s - mean);
    var /= static_cast<double>(samples.size());
    m.stddevMs = std::sqrt(var);
    m.opsPerSecond = mean > 0.0 ? 1000.0 / mean : 0.0;
    (void)symbols;
    return m;
}

}  // namespace mlk
