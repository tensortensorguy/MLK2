// Presburger integer set implementation (see int_set.h).
//
// Key algorithms:
//  - Fourier-Motzkin elimination: equality substitution first (cross
//    multiplication keeps everything integer), then pairwise combination of
//    opposing inequalities via positive multipliers. Every generated row is
//    a non-negative integer combination of source rows, so any integer
//    point of the source satisfies it (soundness).
//  - lexMin/lexMax: lexicographic branch-and-bound over dim values with
//    sound rational pruning and final witness verification (header doc).
#include "mlk/poly/int_set.h"

#include <optional>
#include <utility>

#include "mlk/poly/checked.h"

namespace mlk::poly {

namespace {

/// Floor/ceil integer division (C++ '/' truncates toward zero; these fix
/// the semantics for negative operands — Rule 73: no silent sign bugs).
[[nodiscard]] int64_t floorDiv(int64_t a, int64_t b) noexcept {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

[[nodiscard]] int64_t ceilDiv(int64_t a, int64_t b) noexcept {
    return -floorDiv(-a, b);
}

/// Lexicographic less for dim vectors.
[[nodiscard]] bool lexLess(const SmallVector<int64_t, 8>& a,
                           const SmallVector<int64_t, 8>& b) noexcept {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return a.size() < b.size();
}

/// Multiplies a row by a positive scalar (exact; magnitude checked).
[[nodiscard]] Result<ConstraintRow> scaleRow(const ConstraintRow& row,
                                             int64_t k) noexcept {
    if (k <= 0) {
        return err(ErrorCode::InvalidArgument,
                   "row scaling requires a positive multiplier");
    }
    ConstraintRow r;
    r.isEquality = row.isEquality;
    MLK_TRY_VAR(cst, checked::mulLimited(row.constant, k));
    r.constant = cst;
    for (const int64_t c : row.coeffs) {
        MLK_TRY_VAR(s, checked::mulLimited(c, k));
        r.coeffs.push_back(s);
    }
    return r;
}

/// Adds row b into row a (coefficient-wise; magnitude checked).
[[nodiscard]] Result<ConstraintRow> addRows(const ConstraintRow& a,
                                            const ConstraintRow& b) noexcept {
    ConstraintRow out;
    out.isEquality = a.isEquality && b.isEquality;
    MLK_TRY_VAR(cst, checked::addLimited(a.constant, b.constant));
    out.constant = cst;
    const std::size_t n = a.coeffs.size() < b.coeffs.size()
                              ? a.coeffs.size()
                              : b.coeffs.size();
    for (std::size_t i = 0; i < n; ++i) {
        MLK_TRY_VAR(s, checked::addLimited(a.coeffs[i], b.coeffs[i]));
        out.coeffs.push_back(s);
    }
    return out;
}

/// Subtracts row b from row a (coefficient-wise; magnitude checked).
[[nodiscard]] Result<ConstraintRow> subRows(const ConstraintRow& a,
                                            const ConstraintRow& b) noexcept {
    ConstraintRow out;
    out.isEquality = a.isEquality && b.isEquality;
    MLK_TRY_VAR(cst, checked::subLimited(a.constant, b.constant));
    out.constant = cst;
    const std::size_t n = a.coeffs.size() < b.coeffs.size()
                              ? a.coeffs.size()
                              : b.coeffs.size();
    for (std::size_t i = 0; i < n; ++i) {
        MLK_TRY_VAR(s, checked::subLimited(a.coeffs[i], b.coeffs[i]));
        out.coeffs.push_back(s);
    }
    return out;
}

/// True when two rows are structurally identical.
[[nodiscard]] bool rowsEquivalent(const ConstraintRow& a,
                                  const ConstraintRow& b) noexcept {
    return a.coeffs == b.coeffs && a.constant == b.constant &&
           a.isEquality == b.isEquality;
}

/// Removes duplicate and trivial rows; flags contradictions.
void simplifySystem(Polyhedron& sys) {
    SmallVector<ConstraintRow, 8> kept;
    kept.reserve(sys.rows.size());
    for (ConstraintRow& row : sys.rows) {
        if (row.isTriviallyFalse()) {
            sys.isEmptyFlag = true;
            return;
        }
        if (row.isTriviallyTrue()) continue;
        bool dup = false;
        for (const ConstraintRow& k : kept) {
            if (rowsEquivalent(k, row)) {
                dup = true;
                break;
            }
        }
        if (!dup) kept.push_back(std::move(row));
    }
    sys.rows = std::move(kept);
}

/// FM outcome for one eliminated variable.
struct FmOutcome {
    bool empty{false};
    bool budgetExceeded{false};
    Polyhedron projected{};
};

/// Eliminates ALL occurrences of `var` from `sys` via FM (see file doc).
[[nodiscard]] Result<FmOutcome> fmEliminate(Polyhedron sys, uint32_t var) {
    FmOutcome out;
    out.projected.space = sys.space;

    // Phase 1: substitute equalities containing var into every other row.
    bool substituted = true;
    while (substituted && !sys.isEmptyFlag) {
        substituted = false;
        for (std::size_t ei = 0; ei < sys.rows.size(); ++ei) {
            const ConstraintRow& eq = sys.rows[ei];
            if (!eq.isEquality) continue;
            const int64_t cv = eq.coeffOf(var);
            if (cv == 0) continue;
            // eq: cv*var + R == 0. For any row w: cw*var + W (==/>= 0):
            //   merged = |cv|*w - (cw*sign(cv))*eq  drops the var column
            //   and preserves the solution set exactly (integer-sound).
            //   Negative factors flip the subtraction into an addition.
            const int64_t sign = cv > 0 ? 1 : -1;
            const int64_t absCv = cv > 0 ? cv : -cv;
            for (std::size_t wi = 0; wi < sys.rows.size(); ++wi) {
                if (wi == ei) continue;
                const int64_t cw = sys.rows[wi].coeffOf(var);
                if (cw == 0) continue;
                const int64_t factor = cw * sign;  // merged -= factor*eq
                MLK_TRY_VAR(ws, scaleRow(sys.rows[wi], absCv));
                MLK_TRY_VAR(eqScaled, scaleRow(sys.rows[ei],
                                               factor < 0 ? -factor
                                                          : factor));
                if (factor >= 0) {
                    MLK_TRY_VAR(merged, subRows(ws, eqScaled));
                    sys.rows[wi] = std::move(merged);
                } else {
                    MLK_TRY_VAR(merged, addRows(ws, eqScaled));
                    sys.rows[wi] = std::move(merged);
                }
            }
            // The equality itself is consumed (rational projection keeps
            // the solution set; integer gaps are handled by the documented
            // conservative direction — see int_set.h soundness contract).
            sys.rows[ei].coeffs.clear();
            sys.rows[ei].constant = 0;
            sys.rows[ei].isEquality = false;  // trivially true; removed
            simplifySystem(sys);
            substituted = true;
            break;
        }
    }
    if (sys.isEmptyFlag) {
        out.empty = true;
        return out;
    }

    // Phase 2: pairwise FM over inequalities with opposing var signs.
    SmallVector<ConstraintRow, 8> pos;
    SmallVector<ConstraintRow, 8> neg;
    SmallVector<ConstraintRow, 8> neutral;
    for (ConstraintRow& r : sys.rows) {
        const int64_t cv = r.coeffOf(var);
        if (cv == 0) {
            neutral.push_back(std::move(r));
        } else if (cv > 0) {
            pos.push_back(std::move(r));
        } else {
            neg.push_back(std::move(r));
        }
    }
    for (ConstraintRow& r : neutral) {
        out.projected.addRow(std::move(r));
    }
    if (out.projected.isEmptyFlag) {
        out.empty = true;
        return out;
    }
    for (const ConstraintRow& p : pos) {
        for (const ConstraintRow& n : neg) {
            if (out.projected.rows.size() >= kPolyFmMaxIntermediateRows) {
                out.budgetExceeded = true;
                return out;
            }
            const int64_t cp = p.coeffOf(var);  // > 0
            const int64_t cn = n.coeffOf(var);  // < 0
            // new = p*(-cn) + n*cp: positive multipliers, var cancels.
            MLK_TRY_VAR(ps, scaleRow(p, -cn));
            MLK_TRY_VAR(ns, scaleRow(n, cp));
            MLK_TRY_VAR(merged, addRows(ps, ns));
            out.projected.addRow(std::move(merged));
            if (out.projected.isEmptyFlag) {
                out.empty = true;
                return out;
            }
        }
    }
    simplifySystem(out.projected);
    if (out.projected.isEmptyFlag) out.empty = true;
    return out;
}

/// Full FM feasibility of one system: eliminate every variable (smallest
/// involvement first to slow blow-up).
[[nodiscard]] Result<Feasibility> eliminateAll(Polyhedron sys) {
    simplifySystem(sys);
    if (sys.isEmptyFlag) return Feasibility::Empty;
    if (sys.rowsOverflow) return Feasibility::Unknown;
    for (uint32_t round = 0; round < sys.space.totalVars(); ++round) {
        uint32_t bestVar = 0;
        int64_t bestCount = -1;
        bool picked = false;
        for (uint32_t v = 0; v < sys.space.totalVars(); ++v) {
            int64_t cnt = 0;
            for (const ConstraintRow& r : sys.rows) {
                if (r.coeffOf(v) != 0) ++cnt;
            }
            if (cnt == 0) continue;
            if (!picked || cnt < bestCount) {
                picked = true;
                bestVar = v;
                bestCount = cnt;
            }
        }
        if (!picked) return Feasibility::NonEmpty;  // constant rows only
        MLK_TRY_VAR(fm, fmEliminate(std::move(sys), bestVar));
        if (fm.empty) return Feasibility::Empty;
        if (fm.budgetExceeded) return Feasibility::Unknown;
        sys = std::move(fm.projected);
        simplifySystem(sys);
        if (sys.isEmptyFlag) return Feasibility::Empty;
        if (sys.rowsOverflow) return Feasibility::Unknown;
    }
    for (const ConstraintRow& r : sys.rows) {
        if (r.isTriviallyFalse()) return Feasibility::Empty;
    }
    return Feasibility::NonEmpty;
}

/// Branch-and-bound state for lexMin/lexMax.
struct LexState {
    const Polyhedron* source{nullptr};
    bool maximize{false};
    uint32_t nodeBudget{kPolyLexBbNodeBudget};
    std::optional<SmallVector<int64_t, 8>> best;
};

/// Enumerates candidates in lex order; first integer-verified assignment
/// wins (see int_set.h soundness contract).
///
/// Residual invariant: `residual` references only dims >= prefix.size().
/// Each recursion level SUBSTITUTES x_depth = cand into every row
/// (constant += coeff*cand, coeff := 0) instead of appending an equality
/// row — appending alone leaves the prefix var live in sibling rows, and
/// the univariate interval extraction below would then silently read a
/// coupled row like (i - j >= 0) as (-j >= 0), returning wrong optima
/// (e.g. lexmax{0<=j<=i<=7} = (7,0)). Substitution keeps the interval
/// extraction exact.
[[nodiscard]] Result<bool> lexSearch(LexState& st, const Polyhedron& residual,
                                     SmallVector<int64_t, 8>& prefix) {
    if (st.nodeBudget == 0) {
        return err(ErrorCode::BudgetExceeded,
                   "lexmin/lexmax node budget exhausted");
    }
    --st.nodeBudget;
    const uint32_t depth = static_cast<uint32_t>(prefix.size());
    const uint32_t nDims = st.source->space.nDims;
    if (depth == nDims) {
        if (st.source->containsPoint(prefix)) {
            st.best = prefix;
            return true;  // first verified point is the optimum
        }
        return false;  // rationally feasible, no integer lift at prefix
    }
    // Project out all dims deeper than `depth` so rows only bound x_depth
    // (prefix dims are already substituted away — see the invariant above).
    Polyhedron projected = residual;
    for (uint32_t v = depth + 1; v < nDims; ++v) {
        MLK_TRY_VAR(fm, fmEliminate(std::move(projected), v));
        if (fm.empty) return false;
        if (fm.budgetExceeded) {
            return err(ErrorCode::BudgetExceeded,
                       "lexmin/lexmax FM budget exceeded");
        }
        projected = std::move(fm.projected);
    }
    // Univariate integer interval over dim `depth`. After the projection
    // above every row has support subseteq {depth}, so reading only
    // coeffOf(depth) and r.constant is exact (no hidden prefix terms).
    int64_t lb = INT64_MIN;
    int64_t ub = INT64_MAX;
    for (const ConstraintRow& r : projected.rows) {
        const int64_t c = r.coeffOf(depth);
        if (c == 0) continue;
        if (r.isEquality) {
            // c*x + K == 0 → x == -K/c (infeasible unless divisible)
            if (r.constant % c != 0) return false;
            const int64_t exact = floorDiv(-r.constant, c);
            if (exact < lb || exact > ub) return false;
            lb = exact;
            ub = exact;
            continue;
        }
        if (c > 0) {
            // c*x + K >= 0 → x >= ceil(-K / c)
            const int64_t bound = ceilDiv(-r.constant, c);
            if (bound > lb) lb = bound;
        } else {
            // c*x + K >= 0 (c < 0) → x <= floor(-K / c)
            const int64_t bound = floorDiv(-r.constant, c);
            if (bound < ub) ub = bound;
        }
    }
    if (lb > ub) return false;  // empty interval
    // Only the SEARCH-direction bound must be finite: lexMin walks up from
    // lb (an open upper end just means "enumerate until the witness or the
    // node budget stops us"), lexMax walks down from ub. A missing bound in
    // the search direction means no optimum exists (Rule 67: actionable
    // error, never a wrong optimum).
    if (st.maximize ? ub == INT64_MAX : lb == INT64_MIN) {
        return err(ErrorCode::InvalidArgument,
                   "lexmin/lexmax dimension unbounded in the search "
                   "direction");
    }
    const int64_t start = st.maximize ? ub : lb;
    const int64_t stop = st.maximize ? lb : ub;
    const int64_t step = st.maximize ? -1 : 1;
    for (int64_t cand = start;; cand += step) {
        Polyhedron next = residual;
        // Substitute x_depth = cand into every row: K += c*cand, c := 0.
        // Rows with c == 0 stay; one that became trivially false means the
        // prefix contradicts the system (bail early — sound, since every
        // candidate is re-verified against the source at the leaf anyway).
        for (ConstraintRow& r : next.rows) {
            const int64_t c = r.coeffOf(depth);
            if (c == 0) {
                if (r.isTriviallyFalse()) return false;
                continue;
            }
            int64_t term = 0;
            if (!checked::mul(c, cand, &term) ||
                !checked::add(r.constant, term, &r.constant)) {
                return err(ErrorCode::InvalidArgument,
                           "lexmin/lexmax coefficient overflow while "
                           "substituting a fixed prefix value");
            }
            r.coeffs[depth] = 0;
        }
        prefix.push_back(cand);
        MLK_TRY_VAR(found, lexSearch(st, next, prefix));
        if (found) return true;
        prefix.pop_back();
        if (cand == stop) break;  // sentinel stop => budget-bounded walk
        if (st.nodeBudget == 0) {
            return err(ErrorCode::BudgetExceeded,
                       "lexmin/lexmax node budget exhausted");
        }
    }
    return false;
}

}  // namespace

// --- Polyhedron -------------------------------------------------------------

void Polyhedron::addRow(ConstraintRow row) {
    if (isEmptyFlag) return;
    if (row.coeffs.size() < space.totalVars()) {
        row.coeffs.resize(space.totalVars(), 0);
    }
    if (row.isTriviallyFalse()) {
        isEmptyFlag = true;
        return;
    }
    if (row.isTriviallyTrue()) return;
    if (rows.size() >= kPolyMaxRowsPerSystem) {
        // Rule 10/73: never silently drop constraints; flag the system so
        // feasibility() reports Unknown (conservative direction).
        rowsOverflow = true;
        return;
    }
    rows.push_back(std::move(row));
}

void Polyhedron::addInequality(SmallVector<int64_t, 8> coeffs,
                               int64_t constant) {
    ConstraintRow r;
    r.isEquality = false;
    r.coeffs = std::move(coeffs);
    r.constant = constant;
    addRow(std::move(r));
}

void Polyhedron::addEquality(SmallVector<int64_t, 8> coeffs, int64_t constant) {
    ConstraintRow r;
    r.isEquality = true;
    r.coeffs = std::move(coeffs);
    r.constant = constant;
    addRow(std::move(r));
}

bool Polyhedron::containsPoint(
    const SmallVector<int64_t, 8>& vars) const noexcept {
    if (isEmptyFlag) return false;
    if (vars.size() < space.totalVars()) return false;
    for (const ConstraintRow& r : rows) {
        // Witness evaluation: overflow cannot occur for in-range inputs;
        // on impossible overflow treat the point as violated (conservative
        // for lexmin verification — never accepts a bad witness).
        int64_t acc = r.constant;
        bool overflow = false;
        for (std::size_t i = 0; i < r.coeffs.size(); ++i) {
            if (r.coeffs[i] == 0) continue;
            int64_t term = 0;
            if (!checked::mul(r.coeffs[i], vars[i], &term) ||
                !checked::add(acc, term, &acc)) {
                overflow = true;
                break;
            }
        }
        if (overflow) return false;
        if (r.isEquality) {
            if (acc != 0) return false;
        } else if (acc < 0) {
            return false;
        }
    }
    return true;
}

// --- PresburgerSet ----------------------------------------------------------

PresburgerSet PresburgerSet::empty(const VarSpace& space) noexcept {
    PresburgerSet s;
    s.space = space;
    Polyhedron p;
    p.space = space;
    p.isEmptyFlag = true;
    s.disjuncts.push_back(std::move(p));
    return s;
}

Result<PresburgerSet> PresburgerSet::box(
    const VarSpace& space, const SmallVector<int64_t, 8>& lower,
    const SmallVector<int64_t, 8>& upper) noexcept {
    if (lower.size() < space.nDims || upper.size() < space.nDims) {
        return err(ErrorCode::InvalidArgument,
                   "box: bound vectors shorter than dims");
    }
    PresburgerSet s;
    s.space = space;
    Polyhedron p;
    p.space = space;
    for (uint32_t d = 0; d < space.nDims; ++d) {
        // x_d >= lb  →  x_d - lb >= 0
        ConstraintRow lo;
        lo.isEquality = false;
        lo.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
        lo.coeffs[d] = 1;
        lo.constant = -lower[d];
        p.addRow(std::move(lo));
        // x_d <= ub  →  -x_d + ub >= 0
        ConstraintRow hi;
        hi.isEquality = false;
        hi.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
        hi.coeffs[d] = -1;
        hi.constant = upper[d];
        p.addRow(std::move(hi));
    }
    s.disjuncts.push_back(std::move(p));
    return s;
}

Result<PresburgerSet> PresburgerSet::affineBox(
    const VarSpace& space, const SmallVector<AffineExpr, 8>& lower,
    const SmallVector<AffineExpr, 8>& upper) noexcept {
    if (lower.size() < space.nDims || upper.size() < space.nDims) {
        return err(ErrorCode::InvalidArgument,
                   "affineBox: bound vectors shorter than dims");
    }
    PresburgerSet s;
    s.space = space;
    Polyhedron p;
    p.space = space;
    for (uint32_t d = 0; d < space.nDims; ++d) {
        // x_d - lb(x) >= 0
        {
            const AffineExpr& lb = lower[d];
            if (!lb.space.sameAs(space)) {
                return err(ErrorCode::InvalidArgument,
                           "affineBox: lower bound space mismatch");
            }
            ConstraintRow r;
            r.isEquality = false;
            r.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
            r.coeffs[d] = 1;
            for (std::size_t i = 0; i < space.totalVars(); ++i) {
                MLK_TRY_VAR(t, checked::subLimited(r.coeffs[i],
                                                   lb.coeffs[i]));
                r.coeffs[i] = t;
            }
            MLK_TRY_VAR(loCst,
                        checked::subLimited(0, lb.constant));
            r.constant = loCst;
            p.addRow(std::move(r));
        }
        // ub(x) - x_d >= 0
        {
            const AffineExpr& ub = upper[d];
            if (!ub.space.sameAs(space)) {
                return err(ErrorCode::InvalidArgument,
                           "affineBox: upper bound space mismatch");
            }
            ConstraintRow r;
            r.isEquality = false;
            r.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
            r.coeffs[d] = -1;
            for (std::size_t i = 0; i < space.totalVars(); ++i) {
                MLK_TRY_VAR(t, checked::addLimited(r.coeffs[i],
                                                   ub.coeffs[i]));
                r.coeffs[i] = t;
            }
            r.constant = ub.constant;
            p.addRow(std::move(r));
        }
    }
    s.disjuncts.push_back(std::move(p));
    return s;
}

Result<PresburgerSet> PresburgerSet::intersect(const PresburgerSet& a,
                                               const PresburgerSet& b) noexcept {
    if (!a.space.sameAs(b.space)) {
        return err(ErrorCode::InvalidArgument,
                   "intersect: variable spaces differ");
    }
    PresburgerSet out;
    out.space = a.space;
    for (const Polyhedron& pa : a.disjuncts) {
        if (pa.isEmptyFlag) continue;
        for (const Polyhedron& pb : b.disjuncts) {
            if (pb.isEmptyFlag) continue;
            if (out.disjuncts.size() >= kPolyMaxDisjuncts) {
                return err(ErrorCode::ResourceExhausted,
                           "intersect exceeds disjunct budget");
            }
            Polyhedron p;
            p.space = a.space;
            for (const ConstraintRow& r : pa.rows) p.addRow(r);
            for (const ConstraintRow& r : pb.rows) p.addRow(r);
            simplifySystem(p);
            out.disjuncts.push_back(std::move(p));
        }
    }
    if (out.disjuncts.empty()) {
        Polyhedron p;
        p.space = a.space;
        p.isEmptyFlag = true;
        out.disjuncts.push_back(std::move(p));
    }
    return out;
}

Result<PresburgerSet> PresburgerSet::unite(const PresburgerSet& a,
                                           const PresburgerSet& b) noexcept {
    if (!a.space.sameAs(b.space)) {
        return err(ErrorCode::InvalidArgument,
                   "unite: variable spaces differ");
    }
    PresburgerSet out;
    out.space = a.space;
    for (const Polyhedron& pa : a.disjuncts) {
        if (out.disjuncts.size() >= kPolyMaxDisjuncts) {
            return err(ErrorCode::ResourceExhausted,
                       "unite exceeds disjunct budget");
        }
        out.disjuncts.push_back(pa);
    }
    for (const Polyhedron& pb : b.disjuncts) {
        if (out.disjuncts.size() >= kPolyMaxDisjuncts) {
            return err(ErrorCode::ResourceExhausted,
                       "unite exceeds disjunct budget");
        }
        out.disjuncts.push_back(pb);
    }
    return out;
}

Result<PresburgerSet> PresburgerSet::complement(
    const PresburgerSet& a) noexcept {
    // ~union(P_i) = intersection over i of (~P_i); ~P = union over rows of
    // the negated-row disjuncts.
    PresburgerSet acc;
    acc.space = a.space;
    Polyhedron full;
    full.space = a.space;  // row-free system = universe
    acc.disjuncts.push_back(std::move(full));
    for (const Polyhedron& p : a.disjuncts) {
        if (p.isEmptyFlag) continue;  // complement of empty = universe
        PresburgerSet compP;
        compP.space = a.space;
        if (p.rows.empty()) {
            Polyhedron e;
            e.space = a.space;
            e.isEmptyFlag = true;  // ~universe = empty
            compP.disjuncts.push_back(std::move(e));
        }
        for (const ConstraintRow& r : p.rows) {
            if (compP.disjuncts.size() + 1 > kPolyMaxDisjuncts) {
                return err(ErrorCode::ResourceExhausted,
                           "complement exceeds disjunct budget");
            }
            Polyhedron neg;
            neg.space = a.space;
            if (r.isEquality) {
                // c*x + k == 0  →  (c*x + k >= 1) ∨ (c*x + k <= -1)
                ConstraintRow up;
                up.isEquality = false;
                up.coeffs = r.coeffs;
                up.constant = r.constant - 1;
                neg.addRow(std::move(up));
                ConstraintRow down;
                down.isEquality = false;
                for (const int64_t c : r.coeffs) down.coeffs.push_back(-c);
                down.constant = -r.constant - 1;
                neg.addRow(std::move(down));
            } else {
                // c*x + k >= 0  →  c*x + k <= -1  →  -c*x - k - 1 >= 0
                ConstraintRow nr;
                nr.isEquality = false;
                for (const int64_t c : r.coeffs) nr.coeffs.push_back(-c);
                nr.constant = -r.constant - 1;
                neg.addRow(std::move(nr));
            }
            compP.disjuncts.push_back(std::move(neg));
        }
        MLK_TRY_VAR(next, intersect(acc, compP));
        acc = std::move(next);
    }
    return acc;
}

Result<PresburgerSet> PresburgerSet::subtract(const PresburgerSet& a,
                                              const PresburgerSet& b) noexcept {
    MLK_TRY_VAR(notb, complement(b));
    return intersect(a, notb);
}

Result<Feasibility> PresburgerSet::feasibility() const noexcept {
    bool sawUnknown = false;
    for (const Polyhedron& p : disjuncts) {
        if (p.isEmptyFlag) continue;
        if (p.rowsOverflow) {
            sawUnknown = true;
            continue;
        }
        MLK_TRY_VAR(f, eliminateAll(p));
        if (f == Feasibility::NonEmpty) return Feasibility::NonEmpty;
        if (f == Feasibility::Unknown) sawUnknown = true;
    }
    if (sawUnknown) return Feasibility::Unknown;
    return Feasibility::Empty;
}

Result<bool> PresburgerSet::provablyEmpty() const noexcept {
    MLK_TRY_VAR(f, feasibility());
    return f == Feasibility::Empty;
}

Result<PresburgerSet> PresburgerSet::projectOut(uint32_t var) const noexcept {
    PresburgerSet out;
    out.space = space;
    for (const Polyhedron& p : disjuncts) {
        if (p.isEmptyFlag) continue;
        if (p.rowsOverflow) {
            return err(ErrorCode::ResourceExhausted,
                       "projectOut on over-budget system");
        }
        MLK_TRY_VAR(fm, fmEliminate(p, var));
        if (fm.empty) continue;
        if (fm.budgetExceeded) {
            return err(ErrorCode::ResourceExhausted,
                       "projectOut exceeded FM row budget");
        }
        out.disjuncts.push_back(std::move(fm.projected));
    }
    if (out.disjuncts.empty()) {
        Polyhedron e;
        e.space = space;
        e.isEmptyFlag = true;
        out.disjuncts.push_back(std::move(e));
    }
    return out;
}

Result<PresburgerSet> PresburgerSet::existsProject(
    const SmallVector<bool, 8>& keepDims) const noexcept {
    if (keepDims.size() < space.nDims) {
        return err(ErrorCode::InvalidArgument,
                   "existsProject: keep mask shorter than dims");
    }
    PresburgerSet cur = *this;
    for (uint32_t d = 0; d < space.nDims; ++d) {
        if (d < keepDims.size() && keepDims[d]) continue;
        MLK_TRY_VAR(next, cur.projectOut(d));
        cur = std::move(next);
    }
    return cur;
}

Result<std::optional<SmallVector<int64_t, 8>>> PresburgerSet::lexMin()
    const noexcept {
    if (space.nSyms != 0) {
        return err(ErrorCode::InvalidArgument,
                   "lexMin requires specialized symbols (nSyms == 0)");
    }
    std::optional<SmallVector<int64_t, 8>> best;
    for (const Polyhedron& p : disjuncts) {
        if (p.isEmptyFlag || p.rowsOverflow) continue;
        LexState st;
        st.source = &p;
        st.maximize = false;
        SmallVector<int64_t, 8> prefix;
        MLK_TRY_VAR(found, lexSearch(st, p, prefix));
        if (found && st.best.has_value()) {
            if (!best.has_value() || lexLess(*st.best, *best)) {
                best = st.best;
            }
        }
    }
    return best;
}

Result<std::optional<SmallVector<int64_t, 8>>> PresburgerSet::lexMax()
    const noexcept {
    if (space.nSyms != 0) {
        return err(ErrorCode::InvalidArgument,
                   "lexMax requires specialized symbols (nSyms == 0)");
    }
    std::optional<SmallVector<int64_t, 8>> best;
    for (const Polyhedron& p : disjuncts) {
        if (p.isEmptyFlag || p.rowsOverflow) continue;
        LexState st;
        st.source = &p;
        st.maximize = true;
        SmallVector<int64_t, 8> prefix;
        MLK_TRY_VAR(found, lexSearch(st, p, prefix));
        if (found && st.best.has_value()) {
            if (!best.has_value() || lexLess(*best, *st.best)) {
                best = st.best;
            }
        }
    }
    return best;
}

bool PresburgerSet::containsPoint(
    const SmallVector<int64_t, 8>& vars) const noexcept {
    for (const Polyhedron& p : disjuncts) {
        if (p.containsPoint(vars)) return true;
    }
    return false;
}

std::size_t PresburgerSet::liveDisjuncts() const noexcept {
    std::size_t n = 0;
    for (const Polyhedron& p : disjuncts) {
        if (!p.isEmptyFlag) ++n;
    }
    return n;
}

// --- detail -----------------------------------------------------------------

namespace detail {

Result<std::optional<ConstraintRow>> fmEliminateVar(
    Polyhedron sys, uint32_t var, SmallVector<ConstraintRow, 8>& outRows) {
    MLK_TRY_VAR(fm, fmEliminate(std::move(sys), var));
    if (fm.empty) return std::nullopt;
    if (fm.budgetExceeded) {
        return err(ErrorCode::ResourceExhausted,
                   "fmEliminateVar exceeded FM row budget");
    }
    outRows = std::move(fm.projected.rows);
    return ConstraintRow{};  // presence of a value = system not empty
}

Result<Feasibility> systemFeasibility(const Polyhedron& sys) {
    return eliminateAll(sys);
}

}  // namespace detail

}  // namespace mlk::poly
