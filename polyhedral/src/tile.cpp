// Tiling band detection implementation (see tile.h).
#include "mlk/poly/tile.h"

#include <utility>

#include "schedule_common.h"

namespace mlk::poly {

namespace {

/// True when theta_r varies over the statement's domain (range >= 1).
[[nodiscard]] Result<bool> rowVaries(const Statement& s,
                                     const ScheduleRow& row) {
    // theta_r form over the statement's own dims.
    SmallVector<int64_t, 8> form;
    for (uint32_t d = 0; d < s.space.nDims; ++d) {
        form.push_back(row.stmtCoeffs[s.id][d + 1]);
    }
    const int64_t cst = row.stmtCoeffs[s.id][0];
    Rational lo{};
    Rational hi{};
    bool any = false;
    for (const Polyhedron& dom : s.domain.disjuncts) {
        if (!detail::polyFeasible(dom)) continue;
        MLK_TRY_VAR(mn, detail::minOverPoly(dom, form, cst));
        SmallVector<int64_t, 8> neg;
        for (const int64_t c : form) neg.push_back(-c);
        MLK_TRY_VAR(mx, detail::minOverPoly(dom, neg, -cst));
        // mx = min(-form) = -(max form): negate to get the true maximum.
        Result<Rational> maxForm = ratNeg(mx);
        if (!maxForm.has_value()) {
            return err(ErrorCode::ResourceExhausted,
                       "rowVaries negation overflow");
        }
        if (!any || ratLess(mn, lo)) lo = mn;
        if (!any || ratLess(hi, *maxForm)) hi = *maxForm;
        any = true;
    }
    if (!any) return false;
    MLK_TRY_VAR(range, ratSub(hi, lo));
    return range.isInt() && range.num >= 1;
}

/// Pivot-coordinate distance form for a band row: v_p(dst) - v_p(src)
/// over the product space. Rectangular tiling legality is a condition on
/// LOOP-ITERATION-SPACE distances (the tile function is floor(v_p / t)
/// per band level); theta distances would wrongly let skew coefficients
/// on outer dims leak into the legality decision.
[[nodiscard]] Result<void> pivotDistForm(const Dependence& d,
                                         uint32_t depth, uint32_t pivot,
                                         SmallVector<int64_t, 8>& outCoeffs,
                                         int64_t& outConst) {
    outCoeffs.clear();
    for (uint32_t i = 0; i < depth * 2; ++i) outCoeffs.push_back(0);
    if (pivot >= depth) {
        return err(ErrorCode::InvalidArgument, "pivot dim out of range");
    }
    outCoeffs[pivot] = -1;         // source block: dims [0, depth)
    outCoeffs[depth + pivot] = 1;  // sink block: dims [depth, 2*depth)
    outConst = 0;
    (void)d;
    return {};
}

}  // namespace

Result<TiledInfo> computeTiling(
    const Scop& scop, const SmallVector<Dependence, 16>& deps,
    const PolySchedule& sched, int64_t tileSize) {
    if (tileSize <= 0) {
        return err(ErrorCode::InvalidArgument,
                   "tile size must be positive");
    }
    TiledInfo out;
    if (sched.rows.empty()) return out;  // nothing to tile

    struct DepState {
        PresburgerSet slice{};
        bool resolved{false};
    };
    SmallVector<DepState, 16> states;
    for (const Dependence& d : deps) {
        DepState st;
        st.slice = d.relation;
        st.resolved = false;
        states.push_back(std::move(st));
    }

    for (uint32_t r = 0; r < sched.rows.size(); ++r) {
        if (r >= sched.pivotDim.size()) break;  // legacy schedule guard
        const uint32_t pivot = sched.pivotDim[r];
        SmallVector<uint32_t, 16> live;
        for (uint32_t di = 0; di < states.size(); ++di) {
            if (!states[di].resolved) live.push_back(di);
        }
        // No live deps: further rows order nothing — tiling them is
        // legal but pointless; stop at the meaningful band.
        if (live.empty()) break;

        const ScheduleRow& row = sched.rows[r];
        // Degenerate row (constant over every statement) ends the band.
        bool varies = false;
        for (const Statement& s : scop.statements) {
            MLK_TRY_VAR(v, rowVaries(s, row));
            varies = varies || v;
        }
        if (!varies) break;

        // Band legality (defense in depth): every live dependence's
        // PIVOT-COORDINATE distance must stay >= 0 on its refined slice
        // at this band row — the lex-nonnegative projection condition
        // for rectangular tiling (the scheduler LP guarantees it for
        // rows it emitted; skew-heavy schedules may end the band early).
        SmallVector<SmallVector<int64_t, 8>, 16> distForms;
        SmallVector<int64_t, 16> distConsts;
        for (const uint32_t di : live) {
            const Dependence& d = deps[di];
            SmallVector<int64_t, 8> dc;
            int64_t dk = 0;
            MLK_TRYV(pivotDistForm(d, scop.depth, pivot, dc, dk));
            MLK_TRY_VAR(mn, [&]() -> Result<Rational> {
                Rational best{};
                bool any = false;
                for (const Polyhedron& disjunct :
                     states[di].slice.disjuncts) {
                    if (!detail::polyFeasible(disjunct)) continue;
                    MLK_TRY_VAR(m, detail::minOverPoly(disjunct, dc, dk));
                    if (!any || ratLess(m, best)) best = m;
                    any = true;
                }
                if (!any) {
                    return err(ErrorCode::Internal,
                               "empty slice during tiling analysis");
                }
                return best;
            }());
            if (!(mn.isInt() && mn.num >= 0)) {
                return out;  // band ends before r
            }
            distForms.push_back(std::move(dc));
            distConsts.push_back(dk);
        }

        // Band row accepted; refine slices with dist == 0 for the next
        // row and mark deps resolved at this row (min >= 1).
        out.tileSizes.push_back(tileSize);
        out.bandEnd = r + 1;
        for (uint32_t li = 0; li < live.size(); ++li) {
            const uint32_t di = live[li];
            DepState& st = states[di];
            PresburgerSet next;
            next.space = st.slice.space;
            bool anyLiveDisjunct = false;
            for (const Polyhedron& disjunct : st.slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                Polyhedron updated = disjunct;
                ConstraintRow eq;
                eq.isEquality = true;
                for (const int64_t c : distForms[li]) {
                    eq.coeffs.push_back(c);
                }
                eq.constant = -distConsts[li];
                updated.addRow(std::move(eq));
                if (detail::polyFeasible(updated)) {
                    next.disjuncts.push_back(std::move(updated));
                    anyLiveDisjunct = true;
                }
            }
            if (!anyLiveDisjunct) {
                st.resolved = true;
            } else {
                st.slice = std::move(next);
            }
            // Resolution at this row (min >= 1) — recompute on the
            // ORIGINAL slice before refinement would be needed; Pluto's
            // resolution bookkeeping already ran, so rely on slice
            // emptiness only here.
        }
    }
    out.tiled = out.bandEnd > 0;
    return out;
}

}  // namespace mlk::poly
