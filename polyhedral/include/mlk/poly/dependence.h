// MLK+ polyhedral engine — data dependence analysis.
//
// A dependence is a relation over the product space
//   dims [0, depth) = source instance i, [depth, 2*depth) = sink instance j
// holding exactly the instance pairs that reference the same buffer element
// AND execute in the original program order (source before sink; distinct
// instances for intra-statement dependences). A dependence EXISTS iff its
// relation is not (provably) empty — the tri-state Feasibility contract of
// int_set.h applies (Unknown is kept = conservative, Rules 62/102).
//
// Relations are single polyhedra built from:
//   - source domain rows embedded over the i block,
//   - sink domain rows embedded over the j block,
//   - the flat-index tie row flat_src(i) - flat_dst(j) == 0,
//   - original-order purification: sink >lex source (intra-statement) or
//     sink >=lex source with statement-order tie-break (cross-statement).
// Accumulate stores carry an implicit read access (the accumulator chain),
// recorded by the extractor — that RAW chain is what keeps reduction order
// fixed unless the domain profile allows reassociation (Rules 33/90).
#pragma once

#include <cstdint>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/affine_map.h"
#include "mlk/poly/int_set.h"
#include "mlk/poly/scop.h"

namespace mlk::poly {

/// Dependence budget across the SCoP (Rule 10).
inline constexpr std::size_t kPolyMaxDependences = 64;

enum class DepKind : uint8_t { Raw = 0, War, Waw };

struct Dependence {
    uint32_t srcStmt{0};
    uint32_t dstStmt{0};
    uint32_t srcAccess{0};  // index into src statement's accesses
    uint32_t dstAccess{0};  // index into dst statement's accesses
    DepKind kind{DepKind::Raw};
    /// Product space {2*depth, 0}; empty relation is never stored.
    PresburgerSet relation{};
};

/// Computes all live dependences of the SCoP (RAW/WAR/WAW; RAR excluded).
[[nodiscard]] Result<SmallVector<Dependence, 16>> computeDependences(
    const Scop& scop);

/// True when the source instance must execute before the sink instance is
/// allowed to start (schedule legality query): the relation is non-empty.
[[nodiscard]] bool dependenceLive(const Dependence& dep) noexcept;

}  // namespace mlk::poly
