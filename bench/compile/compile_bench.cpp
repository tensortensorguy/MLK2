// Compile-time benchmark: per-tier pipeline wall time + per-pass phase
// timings for the polyhedral sub-pipeline. Deterministic inputs; medians
// over N reps (Rule 49 protocol adapted to compile time).
//
// Measures:
//   1. scalar graph sin(x^2+3x): tier0/1/2/3 full-pipeline compile time
//   2. GEMM graph (M,K,N = 128): tier2 compile split into
//      analysis / lower / poly sub-phases via kill switches
//   3. tier-1 kernel execution vs interpreter on the scalar graph
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "mlk/autotune/searcher.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/pass/register_all.h"
#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/telemetry.h"

using namespace mlk;

namespace {

double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

double medianOf(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

MathGraph buildScalarGraph(SymbolTable& symbols) {
    GraphBuilder b(symbols);
    const MathType f64 = MathType::scalar(Domain::Float, Dtype::F64);
    const ValueId x = b.placeholder("x", f64);
    const ValueId three = b.constant(3.0, f64);
    auto xx = b.op(MathOp::Mul, {x, x});
    auto tx = b.op(MathOp::Mul, {three, x});
    auto sum = b.op(MathOp::Add, {*xx, *tx});
    auto s = b.op(MathOp::Sin, {*sum});
    b.output(*s);
    return std::move(b.graph());
}

MathGraph buildGemmGraph(SymbolTable& symbols, int64_t m, int64_t k,
                         int64_t n) {
    GraphBuilder b(symbols);
    const MathType t = MathType::tensorValue(Dtype::F64, {m, k});
    const MathType tb = MathType::tensorValue(Dtype::F64, {k, n});
    const ValueId a = b.placeholder("A", t);
    const ValueId bb = b.placeholder("B", tb);
    auto mm = b.op(MathOp::MatMul, {a, bb});
    b.output(*mm);
    return std::move(b.graph());
}

MathDomainProfile scalarProfile(SymbolTable& symbols) {
    MathDomainProfile p;
    p.name = symbols.intern("scalar_f64");
    p.capabilities.set(Capability::HasNumericValues);
    p.capabilities.set(Capability::HasFloatingPoint);
    return p;
}

MathDomainProfile tensorProfile(SymbolTable& symbols) {
    MathDomainProfile p;
    p.name = symbols.intern("tensor_f64_cpu");
    p.capabilities.set(Capability::HasNumericValues);
    p.capabilities.set(Capability::HasFloatingPoint);
    p.capabilities.set(Capability::HasTensorDomain);
    return p;
}

double timeTierCompile(SymbolTable& symbols, const MathGraph& proto,
                       const MathDomainProfile& profile, Tier tier,
                       int reps, uint64_t* hashOut) {
    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) {
        MathGraph graph = proto;  // fresh snapshot per rep (Rule 13)
        TelemetrySink telemetry;
        DiagnosticEngine diag;
        PipelineRunner runner(symbols, &telemetry);
        KernelModule kernel;
        PassContext ctx;
        AccuracyContract contract;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.telemetry = &telemetry;
        ctx.symbols = &symbols;
        ctx.tier = tier;
        ctx.kernelOut = &kernel;
        const double t0 = nowMs();
        auto res = runner.run(tier, ctx, graph, &kernel);
        const double t1 = nowMs();
        if (!res.has_value()) {
            std::printf("  tier%d compile FAILED: %s\n",
                        static_cast<int>(tier),
                        res.error().message.c_str());
            return -1.0;
        }
        if (hashOut != nullptr && r == 0) *hashOut = graphHash(graph);
        samples.push_back(t1 - t0);
    }
    return medianOf(samples);
}

// Phase-split compile for tier2 GEMM: kill switches isolate sub-phases.
double timeTier2Phase(SymbolTable& symbols, const MathGraph& proto,
                      const MathDomainProfile& profile,
                      const std::vector<const char*>& killPasses, int reps) {
    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) {
        MathGraph graph = proto;
        TelemetrySink telemetry;
        DiagnosticEngine diag;
        PipelineRunner runner(symbols, &telemetry);
        KernelModule kernel;
        PassContext ctx;
        AccuracyContract contract;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.telemetry = &telemetry;
        ctx.symbols = &symbols;
        ctx.tier = Tier::Tier2;
        ctx.kernelOut = &kernel;
        OpenHashMap<SymbolId, bool> kills;
        for (const char* kp : killPasses) {
            (void)kills.findOrInsert(symbols.intern(kp), nullptr, true);
        }
        ctx.killSwitches = &kills;
        const double t0 = nowMs();
        auto res = runner.run(Tier::Tier2, ctx, graph, &kernel);
        const double t1 = nowMs();
        if (!res.has_value()) return -1.0;
        samples.push_back(t1 - t0);
    }
    return medianOf(samples);
}

}  // namespace

int main() {
    SymbolTable symbols;
    passes::registerAllPasses(symbols);
    constexpr int kReps = 7;
    const MathDomainProfile sp = scalarProfile(symbols);
    const MathDomainProfile tp = tensorProfile(symbols);

    std::printf("== scalar graph sin(x^2+3x): full-pipeline compile ms ==\n");
    MathGraph scalarProto = buildScalarGraph(symbols);
    for (const Tier tier :
         {Tier::Tier0, Tier::Tier1, Tier::Tier2, Tier::Tier3}) {
        const double ms = timeTierCompile(symbols, scalarProto, sp, tier,
                                          kReps, nullptr);
        std::printf("tier%d: %.3f ms (median of %d)\n",
                    static_cast<int>(tier), ms, kReps);
    }

    std::printf(
        "== GEMM 128^3 (tensor_f64_cpu): tier2 compile phases ms ==\n");
    MathGraph gemmProto = buildGemmGraph(symbols, 128, 128, 128);
    const double full = timeTierCompile(symbols, gemmProto, tp, Tier::Tier2,
                                        kReps, nullptr);
    std::printf("tier2 full: %.3f ms\n", full);
    struct PhaseDef {
        const char* name;
        std::vector<const char*> kills;
    };
    const PhaseDef phases[] = {
        {"no-poly (analysis+lower)",
         {"poly.synth", "poly.scop_detect", "poly.dependence",
          "poly.schedule", "poly.tile", "poly.codegen", "poly.verify"}},
        {"no-schedule+tile+codegen",
         {"poly.schedule", "poly.tile", "poly.codegen"}},
        {"no-verify", {"poly.verify"}},
    };
    for (const PhaseDef& ph : phases) {
        const double ms = timeTier2Phase(symbols, gemmProto, tp, ph.kills,
                                         kReps);
        std::printf("  %s: %.3f ms\n", ph.name, ms);
    }

    std::printf("== scalar run: interpreter vs tier1 kernel, us ==\n");
    MathGraph scalarCompiled = buildScalarGraph(symbols);
    TelemetrySink telemetry;
    ExecutionEngine engine(symbols, telemetry);
    AccuracyContract contract;
    SmallVector<double, 8> inputs{2.0};
    std::vector<double> interp;
    std::vector<double> kernd;
    MathDomainProfile p2 = scalarProfile(symbols);
    for (int r = 0; r < 2001; ++r) {
        const double t0 = nowMs();
        auto a = interpretGraph(scalarCompiled, inputs);
        const double t1 = nowMs();
        if (!a.has_value()) return 1;
        if (r >= 200) interp.push_back(t1 - t0);
        if (r % 2 == 0) continue;  // alternate emphasis
        const double t2 = nowMs();
        auto k = engine.execute(scalarCompiled, p2, contract, Tier::Tier1,
                                inputs);
        const double t3 = nowMs();
        if (!k.has_value()) return 1;
        if (r >= 200) kernd.push_back(t3 - t2);
    }
    std::printf("interpretGraph: %.4f us (median)\n",
                medianOf(interp) * 1000.0);
    std::printf("tier1 execute : %.4f us (median; includes kernel dispatch "
                "on the pre-installed realization)\n",
                medianOf(kernd) * 1000.0);
    return 0;
}
