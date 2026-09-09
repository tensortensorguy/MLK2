// Runtime implementations: telemetry, GraphState, fallback, cache.
#include "mlk/runtime/telemetry.h"
#include "mlk/runtime/graph_state.h"
#include "mlk/runtime/fallback.h"
#include "mlk/runtime/buffer.h"
#include "mlk/runtime/cache.h"

namespace mlk {

// --- Telemetry ------------------------------------------------------------
const char* telemetryEventName(TelemetryEventKind k) noexcept {
    switch (k) {  // Rule 78: exhaustive
        case TelemetryEventKind::CompileAttempt: return "compile_attempt";
        case TelemetryEventKind::CompileFailure: return "compile_failure";
        case TelemetryEventKind::TuneAttempt: return "tune_attempt";
        case TelemetryEventKind::TuneFailure: return "tune_failure";
        case TelemetryEventKind::Fallback: return "fallback";
        case TelemetryEventKind::GuardFailure: return "guard_failure";
        case TelemetryEventKind::Invalidation: return "invalidation";
        case TelemetryEventKind::BudgetViolation: return "budget_violation";
        case TelemetryEventKind::CachePressure: return "cache_pressure";
        case TelemetryEventKind::Blacklist: return "blacklist";
        case TelemetryEventKind::TierTransition: return "tier_transition";
        case TelemetryEventKind::PerfCounter: return "perf_counter";
    }
    return "?";
}

void TelemetrySink::record(TelemetryEvent e) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() >= constants::kTelemetryRingCapacity) {
        events_.erase(events_.begin());  // ring drop-oldest
    }
    events_.push_back(e);
}

std::size_t TelemetrySink::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
}

TelemetryEvent TelemetrySink::at(std::size_t i) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_[i];
}

uint32_t TelemetrySink::countKind(TelemetryEventKind k) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t count = 0;
    for (const auto& e : events_) {
        if (e.kind == k) ++count;
    }
    return count;
}

json::Value TelemetrySink::toJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    json::Value arr = json::Array{};
    for (const auto& e : events_) {
        json::Value eo = json::Object{};
        eo.set("kind", json::Value{telemetryEventName(e.kind)});
        eo.set("counter", json::Value{
                              static_cast<int64_t>(e.counter)});
        eo.set("from_tier", json::Value{static_cast<int64_t>(e.fromTier)});
        eo.set("to_tier", json::Value{static_cast<int64_t>(e.toTier)});
        eo.set("graph_version",
               json::Value{static_cast<int64_t>(e.graphVersion)});
        arr.push(std::move(eo));
    }
    return arr;
}

// --- GraphState -------------------------------------------------------------
json::Value GraphState::toJson() const {
    json::Value doc = json::Object{};
    doc.set("resume_node",
            json::Value{static_cast<int64_t>(resumeNode)});
    doc.set("graph_version",
            json::Value{static_cast<int64_t>(graphVersion)});
    doc.set("domain_version",
            json::Value{static_cast<int64_t>(domainVersion)});
    json::Value binds = json::Array{};
    for (const auto& b : bindings) {
        json::Value bo = json::Object{};
        bo.set("value", json::Value{static_cast<int64_t>(b.value)});
        bo.set("is_int", json::Value{b.isInt});
        if (b.isInt) {
            bo.set("i64", json::Value{b.i64});
        } else {
            bo.set("f64", json::Value{b.f64});
        }
        binds.push(std::move(bo));
    }
    doc.set("bindings", std::move(binds));
    return doc;
}

Result<GraphState> GraphState::fromJson(const json::Value& doc) {
    GraphState s;
    const json::Value* rn = doc.find("resume_node");
    if (rn == nullptr || !rn->isInt() || rn->asInt() < 0) {
        return err(ErrorCode::ParseError, "GraphState missing resume_node",
                   101);
    }
    s.resumeNode = static_cast<NodeId>(rn->asInt());
    const json::Value* gv = doc.find("graph_version");
    if (gv != nullptr && gv->isInt()) s.graphVersion = static_cast<uint64_t>(gv->asInt());
    const json::Value* dv = doc.find("domain_version");
    if (dv != nullptr && dv->isInt()) s.domainVersion = static_cast<uint64_t>(dv->asInt());
    const json::Value* binds = doc.find("bindings");
    if (binds == nullptr || !binds->isArray()) {
        return err(ErrorCode::ParseError, "GraphState missing bindings", 101);
    }
    for (const auto& b : binds->asArray()) {
        ValueBinding vb;
        const json::Value* v = b.find("value");
        if (v == nullptr || !v->isInt()) {
            return err(ErrorCode::ParseError, "binding missing value id", 101);
        }
        vb.value = static_cast<ValueId>(v->asInt());
        const json::Value* isInt = b.find("is_int");
        vb.isInt = isInt != nullptr && isInt->isBool() && isInt->asBool();
        if (vb.isInt) {
            const json::Value* iv = b.find("i64");
            if (iv == nullptr || !iv->isInt()) {
                return err(ErrorCode::ParseError, "int binding missing i64",
                           101);
            }
            vb.i64 = iv->asInt();
        } else {
            const json::Value* fv = b.find("f64");
            if (fv == nullptr || !fv->isNumber()) {
                return err(ErrorCode::ParseError, "fp binding missing f64",
                           101);
            }
            vb.f64 = fv->asDouble();
        }
        s.bindings.push_back(vb);
    }
    // Rule 101: the verifier rejects incomplete GraphState — an empty
    // binding list with a mid-graph resume position is invalid.
    if (s.bindings.empty() && s.resumeNode != constants::kInvalidId &&
        s.resumeNode != 0) {
        return err(ErrorCode::InvalidArtifact,
                   "GraphState incomplete: mid-graph resume without bindings",
                   101);
    }
    return s;
}

// --- Fallback -----------------------------------------------------------------
const char* memorySpaceName(MemorySpace s) noexcept {
    switch (s) {  // Rule 78: exhaustive
        case MemorySpace::Register: return "register";
        case MemorySpace::Scratch: return "scratch";
        case MemorySpace::Shared: return "shared";
        case MemorySpace::L1: return "l1";
        case MemorySpace::L2: return "l2";
        case MemorySpace::DRAM: return "dram";
        case MemorySpace::HBM: return "hbm";
        case MemorySpace::Host: return "host";
        case MemorySpace::Persistent: return "persistent";
    }
    return "?";
}

GraphState FallbackEngine::capture(const InterpreterState& state) const {
    GraphState s;
    s.resumeNode = state.resumeNode;
    s.graphVersion = state.graphVersion;
    state.scalars.forEach([&](ValueId v, double d) {
        ValueBinding b;
        b.value = v;
        b.f64 = d;
        b.isInt = false;
        s.bindings.push_back(b);
    });
    return s;
}

Result<InterpreterState> FallbackEngine::reconstruct(
    const GraphState& snapshot) const {
    InterpreterState st;
    st.resumeNode = snapshot.resumeNode;
    st.graphVersion = snapshot.graphVersion;
    for (const auto& b : snapshot.bindings) {
        double* slot = st.scalars.findOrInsert(b.value, nullptr, 0.0);
        *slot = b.isInt ? static_cast<double>(b.i64) : b.f64;
    }
    return st;
}

void FallbackEngine::recordFallback(SymbolId site) {
    uint32_t* slot = fallbackCounts_.findOrInsert(site, nullptr, 0);
    ++*slot;
    TelemetryEvent e;
    e.kind = TelemetryEventKind::Fallback;
    e.pass = site;
    e.counter = *slot;
    telemetry_.record(e);
}

bool FallbackEngine::siteThrottled(SymbolId site) const {
    const uint32_t* count = fallbackCounts_.find(site);
    return count != nullptr &&
           *count >= constants::kFallbackThrottleThreshold;
}

// --- Realization cache -------------------------------------------------------
json::Value CacheKey::toJson() const {
    json::Value v = json::Object{};
    v.set("graph_hash", json::Value{static_cast<int64_t>(graphHash)});
    v.set("property_facts_hash",
          json::Value{static_cast<int64_t>(propertyFactsHash)});
    v.set("shape_bucket", json::Value{shapeBucket});
    v.set("accuracy_hash", json::Value{static_cast<int64_t>(accuracyHash)});
    v.set("profile_version", json::Value{static_cast<int64_t>(profileVersion)});
    v.set("compiler_version",
          json::Value{static_cast<int64_t>(compilerVersion)});
    v.set("pass_pipeline_hash",
          json::Value{static_cast<int64_t>(passPipelineHash)});
    v.set("superopt_version",
          json::Value{static_cast<int64_t>(superoptVersion)});
    v.set("hardware_fingerprint",
          json::Value{static_cast<int64_t>(hardwareFingerprint)});
    return v;
}

Result<CacheKey> CacheKey::fromJson(const json::Value& doc) {
    CacheKey k;
    auto getH = [&](const char* f, HashValue& out) -> Result<Ok> {
        const json::Value* v = doc.find(f);
        if (v == nullptr || !v->isInt()) {
            return err(ErrorCode::ParseError,
                       std::string("cache key missing ") + f, 57);
        }
        out = static_cast<HashValue>(v->asInt());
        return ok();
    };
    MLK_TRYV(getH("graph_hash", k.graphHash));
    MLK_TRYV(getH("property_facts_hash", k.propertyFactsHash));
    MLK_TRYV(getH("accuracy_hash", k.accuracyHash));
    MLK_TRYV(getH("pass_pipeline_hash", k.passPipelineHash));
    MLK_TRYV(getH("hardware_fingerprint", k.hardwareFingerprint));
    const json::Value* bucket = doc.find("shape_bucket");
    if (bucket != nullptr && bucket->isInt()) k.shapeBucket = bucket->asInt();
    const json::Value* pv = doc.find("profile_version");
    if (pv != nullptr && pv->isInt()) {
        k.profileVersion = static_cast<uint32_t>(pv->asInt());
    }
    const json::Value* cv = doc.find("compiler_version");
    if (cv != nullptr && cv->isInt()) {
        k.compilerVersion = static_cast<uint32_t>(cv->asInt());
    }
    return k;
}

json::Value CacheEntry::toJson(SymbolTable& symbols) const {
    json::Value v = json::Object{};
    v.set("format_version",
          json::Value{static_cast<int64_t>(formatVersion)});
    v.set("key", key.toJson());
    v.set("strategy", json::Value{symbols.text(strategy)});
    v.set("kernel_hash", json::Value{static_cast<int64_t>(kernelHash)});
    v.set("schedule", schedule);
    if (proofRef != kInvalidSymbolId) {
        v.set("proof_ref", json::Value{symbols.text(proofRef)});
    }
    v.set("measured_ms", json::Value{measuredMs});
    return v;
}

Result<CacheEntry> CacheEntry::fromJson(const json::Value& doc,
                                        SymbolTable& symbols) {
    CacheEntry e;
    const json::Value* fv = doc.find("format_version");
    if (fv == nullptr || !fv->isInt() ||
        fv->asInt() !=
            static_cast<int64_t>(constants::kRealizationCacheFormatVersion)) {
        return err(ErrorCode::InvalidArtifact,
                   "cache entry format version mismatch (Rule 37)", 37);
    }
    const json::Value* keyv = doc.find("key");
    if (keyv == nullptr) {
        return err(ErrorCode::InvalidArtifact, "cache entry missing key", 57);
    }
    MLK_TRY_VAR(k, CacheKey::fromJson(*keyv));
    e.key = k;
    const json::Value* st = doc.find("strategy");
    if (st != nullptr && st->isString()) {
        e.strategy = symbols.intern(st->asString());
    }
    const json::Value* kh = doc.find("kernel_hash");
    if (kh != nullptr && kh->isInt()) {
        e.kernelHash = static_cast<HashValue>(kh->asInt());
    }
    const json::Value* sched = doc.find("schedule");
    if (sched != nullptr) e.schedule = *sched;
    const json::Value* mm = doc.find("measured_ms");
    if (mm != nullptr && mm->isNumber()) e.measuredMs = mm->asDouble();
    return e;
}

const CacheEntry* RealizationCache::find(const CacheKey& key) const {
    return entries_.find(key.totalHash());
}

void RealizationCache::store(const CacheEntry& entry) {
    CacheEntry copy = entry;
    CacheEntry* slot =
        entries_.findOrInsert(entry.key.totalHash(), nullptr, copy);
    *slot = copy;
}

std::size_t RealizationCache::size() const { return entries_.size(); }

void RealizationCache::clear() { entries_.clear(); }

json::Value RealizationCache::toJson() const {
    json::Value arr = json::Array{};
    entries_.forEach([&](HashValue, const CacheEntry& e) {
        arr.push(e.toJson(symbols_));
    });
    return arr;
}

}  // namespace mlk
