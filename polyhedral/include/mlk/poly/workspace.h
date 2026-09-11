// MLK+ polyhedral engine — per-compilation polyhedral pipeline workspace.
//
// PolyWorkspace is the explicit state shared by the poly.* pass family
// through PassContext (Rule 144: no hidden globals). The embedding runner
// allocates it before the pipeline and frees it after; passes only mutate
// the sections they own and reset the downstream validity flags.
#pragma once

#include "mlk/poly/dependence.h"
#include "mlk/poly/pluto.h"
#include "mlk/poly/tile.h"
#include "mlk/poly/scop.h"

namespace mlk::poly {

/// Shared per-compilation state handed to poly.* passes through
/// PassContext (explicit workspace, no hidden globals — Rule 144).
struct PolyWorkspace {
    Scop scop{};
    bool scopValid{false};
    SmallVector<Dependence, 16> dependences{};
    bool dependencesValid{false};
    PolySchedule schedule{};
    TiledInfo tiled{};
    KernelModule baselineKernel{};  // pre-codegen copy (verify restores)
    bool baselineSaved{false};
    bool scheduleValid{false};
    bool tileValid{false};
    bool codegenValid{false};
    // Counters surfaced through telemetry (Rule 138).
    uint32_t statsStatements{0};
    uint32_t statsDependences{0};
    uint32_t statsScheduleRows{0};
    uint32_t statsTiledBands{0};
};

[[nodiscard]] PolyWorkspace* createPolyWorkspace();
void destroyPolyWorkspace(PolyWorkspace* ws) noexcept;

}  // namespace mlk::poly
