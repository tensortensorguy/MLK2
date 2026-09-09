// MLK+ event sink interface (Rule 30/157): cross-layer telemetry without
// layering violations. The runtime TelemetrySink implements this; compiler
// layers see only the interface (dependency flow, layout spec §3).
#pragma once

#include "mlk/core/symbol_table.h"

namespace mlk {

class IEventSink {
public:
    virtual ~IEventSink() = default;
    /// kind/event names are owned by the runtime telemetry schema; the
    /// compiler layer passes stable small ids.
    virtual void event(uint32_t kind, SymbolId pass, SymbolId reason,
                       uint64_t counter) noexcept = 0;
};

}  // namespace mlk
