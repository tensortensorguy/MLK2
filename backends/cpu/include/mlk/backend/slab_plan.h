// MLK+ shared-memory slab planning (docs/polyhedral_spec.md
// #GPU-backend, round 21). A SLAB is a read-only copy of a buffer's
// reachable element range placed in on-device shared memory (or a
// heap scratch block in the C++ mirror artifact) for the duration of
// ONE root's execution: every read of that buffer inside the root is
// redirected to the slab copy, so the block's threads hit shared
// memory instead of re-reading the same global lines.
//
// Soundness contract (why the copy is invisible):
//   1. READ-ONLY PROOF (buffer granularity, module-wide): a buffer
//      qualifies only if NO Store node in the whole module targets it
//      (temps are stored by construction and never qualify). Its
//      global values therefore cannot change between the cooperative
//      load and any redirected read.
//   2. VALUE IDENTITY: a redirected read at index form F returns the
//      value loaded from B[F] — the slab position is (F - hullMin),
//      and the load fills exactly the positions whose global index
//      lands in [hullMin, hullMax] and inside the buffer. Every LIVE
//      read has an in-bounds form value (walker parity), so every
//      redirected read hits a loaded position. Hull-excess positions
//      (padding) are skipped by the load guard and never read.
//   3. HULL SUPERSET: the reachable index range is the exact interval
//      arithmetic hull of all the buffer's read forms over the var
//      hulls (per-var interval hulls of the loop bounds, padded forms
//      included) — a superset of every value any thread can read.
//      Hull bound texts expand to constants and dims reads only, so
//      the slab size is host-computable (the runtime pick) and
//      device-computable (the load) from the SAME texts.
//
// Exactness: the redirect changes WHERE a value is read from, never
// WHICH value or in WHAT order the arithmetic consumes it — the
// bit-exactness policy of the artifact is unchanged. The C++ mirror
// artifact (same plan, heap scratch instead of __shared__) is
// differentially tested bit-exact against the walker; the CUDA text
// is structure-verified (no nvcc in the verification environment —
// declared boundary, spec #GPU-backend).
//
// Runtime pick: dims are runtime values, so "does the slab fit the
// budget?" is a RUNTIME question. The CUDA wrapper emits BOTH forms
// (slab twin + plain twin) and picks per launch: the slab twin only
// when the runtime byte size is within the declared budget (and
// non-negative), the plain form otherwise. Both are correct; the
// choice is recorded in the generated header (Rule 148), never
// silent.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlk/core/result.h"
#include "mlk/support/kernel_ir.h"

namespace mlk {

/// Emitter options for the slab mechanism (opt-in; the default value
/// reproduces the historical emission byte-for-byte).
struct SlabEmitOptions {
    /// Enable slab planning + dual-form emission (CUDA) / preload
    /// blocks (C++ mirror).
    bool sharedMemSlabs{false};
    /// Declared per-launch dynamic shared-memory budget in bytes. The
    /// generated wrapper picks the slab twin only when the runtime
    /// slab size fits; otherwise it launches the plain form. The
    /// default is the 48 KiB static allocation guarantee every
    /// sm_70+ device makes for dynamic shared memory.
    int64_t smemBudgetBytes{49152};
};

/// One planned slab: the buffer's reachable range [min, min + span)
/// as host-computable texts (constants and dims reads only).
struct Slab {
    uint32_t buffer{0};
    /// Inclusive hull minimum of the buffer's read forms (text).
    std::string minText{};
    /// hullMax - hullMin + 1 (text); may evaluate <= 0 when every
    /// read form's var hull is statically empty (then no live read
    /// exists and the slab is simply unused).
    std::string spanText{};
    /// The buffer's total element count (dims product, text) — the
    /// load's in-bounds guard.
    std::string elemsText{};
};

/// Per-root plan: qualifying slabs (buffer-id order) + recorded skips
/// (Rule 148: a buffer with reads that failed the read-only proof is
/// RECORDED, never silently dropped).
struct RootSlabPlan {
    std::vector<Slab> slabs{};
    std::vector<std::string> notes{};
};

/// Plans slabs for the given roots (id order). `stored[bid]` must be
/// the module-wide store-target table (the emitters' own analysis);
/// `dimsName[bid]` the caller's dims-text table so the hull texts use
/// exactly the names the surrounding artifact uses. Pure function of
/// the module — re-planning is byte-identical (determinism).
[[nodiscard]] Result<std::vector<RootSlabPlan>> planRootSlabs(
    const KernelModule& kernel, const std::vector<uint32_t>& roots,
    const std::vector<std::string>& dimsName,
    const std::vector<bool>& stored, const SlabEmitOptions& opts);

}  // namespace mlk
