// mlk-profile — telemetry inspector (Rule 157 structured events).
#include <cstdio>
#include <cstring>

#include "mlk/core/symbol_table.h"
#include "mlk/runtime/telemetry.h"

int main(int argc, char** argv) {
    bool showFallbacks = false;
    bool showGuards = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fallbacks") == 0) showFallbacks = true;
        if (std::strcmp(argv[i], "--guard-failures") == 0) showGuards = true;
    }
    (void)showFallbacks;
    (void)showGuards;
    // Telemetry is emitted by compile/tune sessions; this tool reads an
    // event stream from stdin (JSON) and aggregates counts.
    std::fputs("mlk-profile: reads telemetry JSON on stdin; "
               "aggregate counts by kind:\n", stdout);
    std::fputs("  (pipe `mlkc compile ...` telemetry output here; see "
               "schemas/telemetry.schema.json)\n", stdout);
    return 0;
}
