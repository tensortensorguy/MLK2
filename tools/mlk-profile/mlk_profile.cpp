// mlk-profile — telemetry inspector (Rule 157 structured events).
//
// Reads the telemetry JSON stream on stdin (the array produced by
// `mlkc compile ... --telemetry`, schemas/telemetry.schema.json) and
// aggregates it: per-kind event counts and counter sums, deterministic
// order. Filters:
//   --fallbacks       list fallback events with pass/reason/counter
//   --guard-failures  list guard_failure events with pass/reason/counter
// Exit codes: 0 ok, 2 malformed input / unreadable stream.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "mlk/support/json.h"

namespace {

bool readAllStdin(std::string& out) {
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) out.append(buf, n);
    return !std::ferror(stdin);
}

/// Event objects tolerate the schema's optionality: kind is required,
/// everything else is read when present.
struct EventView {
    std::string kind;
    std::string pass;
    std::string reason;
    long long counter = 0;
};

bool asEvent(const mlk::json::Value& v, EventView& out) {
    if (!v.isObject()) return false;
    const mlk::json::Value* kind = v.find("kind");
    if (kind == nullptr || !kind->isString()) return false;
    out.kind = kind->asString();
    if (const mlk::json::Value* p = v.find("pass");
        p != nullptr && p->isString()) {
        out.pass = p->asString();
    }
    if (const mlk::json::Value* r = v.find("reason");
        r != nullptr && r->isString()) {
        out.reason = r->asString();
    }
    if (const mlk::json::Value* c = v.find("counter");
        c != nullptr && c->isNumber()) {
        out.counter = c->asInt();
    }
    return true;
}

void listKind(const std::string& kind, const std::string& label,
              const std::vector<EventView>& events) {
    std::printf("%s:\n", label.c_str());
    std::size_t shown = 0;
    for (const EventView& e : events) {
        if (e.kind != kind) continue;
        std::printf("  pass=%s reason=%s counter=%lld\n",
                    e.pass.empty() ? "-" : e.pass.c_str(),
                    e.reason.empty() ? "-" : e.reason.c_str(), e.counter);
        ++shown;
    }
    if (shown == 0) std::printf("  (none)\n");
}
}  // namespace

int main(int argc, char** argv) {
    bool showFallbacks = false;
    bool showGuards = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fallbacks") == 0) showFallbacks = true;
        if (std::strcmp(argv[i], "--guard-failures") == 0) showGuards = true;
    }

    std::string text;
    if (!readAllStdin(text)) {
        std::fprintf(stderr, "mlk-profile: stdin read failed\n");
        return 2;
    }
    if (text.find_first_not_of(" \t\r\n") == std::string::npos) {
        std::fprintf(stderr, "mlk-profile: empty telemetry stream\n");
        return 2;
    }
    mlk::json::Value doc;
    {
        auto parsed = mlk::json::parse(text);
        if (!parsed.has_value()) {
            std::fprintf(stderr, "mlk-profile: parse error: %s\n",
                         parsed.error().message.c_str());
            return 2;
        }
        doc = std::move(*parsed);
    }
    // Accept a bare event array, or an object carrying one under "events".
    const mlk::json::Value* arr = nullptr;
    if (doc.isArray()) {
        arr = &doc;
    } else if (doc.isObject()) {
        arr = doc.find("events");
    }
    if (arr == nullptr || !arr->isArray()) {
        std::fprintf(stderr,
                     "mlk-profile: expected an event array "
                     "(or {\"events\": [...]}) on stdin\n");
        return 2;
    }

    std::vector<EventView> events;
    events.reserve(arr->asArray().size());
    for (const mlk::json::Value& v : arr->asArray()) {
        EventView e;
        if (!asEvent(v, e)) {
            std::fprintf(stderr, "mlk-profile: malformed event entry\n");
            return 2;
        }
        events.push_back(std::move(e));
    }

    // Aggregate: count + counter sum per kind (std::map = deterministic
    // sorted output, Rule 143).
    std::map<std::string, std::pair<std::size_t, long long>> agg;
    for (const EventView& e : events) {
        auto& slot = agg[e.kind];
        slot.first += 1;
        slot.second += e.counter;
    }
    std::printf("mlk-profile: %zu events\n", events.size());
    for (const auto& [kind, stat] : agg) {
        std::printf("  %-16s count=%zu counter_sum=%lld\n", kind.c_str(),
                    stat.first, stat.second);
    }
    if (showFallbacks) listKind("fallback", "fallbacks", events);
    if (showGuards) listKind("guard_failure", "guard failures", events);
    return 0;
}
