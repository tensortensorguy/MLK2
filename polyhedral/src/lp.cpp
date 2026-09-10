// Exact rational simplex implementation (see lp.h).
//
// Tableau layout
//   columns: [ x+ (n) | x- (n) | surplus/slack (nGe) | artificial (nArt) ]
//   rows:    one per constraint; rhs kept in a parallel vector.
// Free program variables are split (x = x+ - x-, both >= 0). Equality and
// inequality rows are normalized so the rhs is non-negative BEFORE the
// artificial column (+1, the initial basis) is attached. Phase 1 minimizes
// the artificial sum (infeasible iff the optimum is > 0); Phase 2
// minimizes the real objective with artificial columns barred from
// entering. Bland's rule everywhere: lowest index for entering (negative
// reduced cost) and lowest basis column for ratio ties — deterministic and
// cycling-free (Rule 53).
#include "mlk/poly/lp.h"

#include <utility>

namespace mlk::poly {

namespace {

class Simplex {
public:
    [[nodiscard]] bool build(const LinearProgram& lp) {
        const uint32_t n = lp.nVars;
        const uint32_t nEq = static_cast<uint32_t>(lp.eqRows.size());
        const uint32_t nGe = static_cast<uint32_t>(lp.geRows.size());
        nVars_ = n;
        nStruct_ = n * 2;
        nSurplus_ = nGe;
        nArt_ = nEq + nGe;
        nCols_ = nStruct_ + nSurplus_ + nArt_;
        if (nStruct_ == 0 || nCols_ > kLpMaxVars) return false;
        m_ = nEq + nGe;
        if (m_ > kLpMaxRows) return false;

        rows_.clear();
        rhs_.clear();
        basis_.clear();
        uint32_t artIdx = 0;
        // Equality rows.
        for (uint32_t r = 0; r < nEq; ++r) {
            SmallVector<Rational, 16> row(nCols_, Rational{});
            if (!fillStructural(lp.eqRows[r], row)) return false;
            Rational b = r < lp.eqRhs.size() ? lp.eqRhs[r] : Rational{};
            if (b.sign() < 0) {
                if (!negateRow(row)) return false;
                if (!negateInto(b, b)) return false;
            }
            row[nStruct_ + nSurplus_ + artIdx] = Rational{1, 1};
            rhs_.push_back(b);
            basis_.push_back(nStruct_ + nSurplus_ + artIdx);
            ++artIdx;
            rows_.push_back(std::move(row));
        }
        // Inequality rows (>=): A x - s (+a) = b.
        for (uint32_t r = 0; r < nGe; ++r) {
            SmallVector<Rational, 16> row(nCols_, Rational{});
            if (!fillStructural(lp.geRows[r], row)) return false;
            Rational b = r < lp.geRhs.size() ? lp.geRhs[r] : Rational{};
            const bool flipped = b.sign() < 0;
            if (flipped) {
                if (!negateRow(row)) return false;
                if (!negateInto(b, b)) return false;
            }
            row[nStruct_ + r] = flipped ? Rational{1, 1} : Rational{-1, 1};
            row[nStruct_ + nSurplus_ + artIdx] = Rational{1, 1};
            rhs_.push_back(b);
            basis_.push_back(nStruct_ + nSurplus_ + artIdx);
            ++artIdx;
            rows_.push_back(std::move(row));
        }
        return artIdx == nArt_;
    }

    /// Phase 1: true = feasible basis found (artificial sum == 0).
    [[nodiscard]] Result<bool> phase1() {
        SmallVector<Rational, 16> cost(nCols_, Rational{});
        for (uint32_t a = 0; a < nArt_; ++a) {
            cost[nStruct_ + nSurplus_ + a] = Rational{1, 1};
        }
        MLK_TRYV(runSimplex(cost, nCols_));
        Rational sum{};
        for (uint32_t r = 0; r < m_; ++r) {
            if (basis_[r] >= nStruct_ + nSurplus_) {
                MLK_TRY_VAR(added, ratAdd(sum, rhs_[r]));
                sum = added;
            }
        }
        return sum.isZero();
    }

    /// Phase 2: minimize c.x; fills the objective and primal solution.
    [[nodiscard]] Result<LpStatus> phase2(const SmallVector<Rational, 8>& c,
                                          Rational& outObj,
                                          SmallVector<Rational, 8>& outX) {
        SmallVector<Rational, 16> cost(nCols_, Rational{});
        for (uint32_t k = 0; k * 2 < nStruct_ && k < c.size(); ++k) {
            cost[2 * k] = c[k];
            Result<Rational> neg = ratNeg(c[k]);
            if (!neg.has_value()) {
                return err(ErrorCode::ResourceExhausted,
                           "lp objective overflow");
            }
            cost[2 * k + 1] = *neg;
        }
        // Artificial columns never enter in phase 2.
        MLK_TRY_VAR(st, runSimplex(cost, nStruct_ + nSurplus_));
        if (st != LpStatus::Optimal) return st;
        MLK_TRY_VAR(xv, extractX());
        Rational acc{};
        for (uint32_t k = 0; k < c.size() && k < xv.size(); ++k) {
            MLK_TRY_VAR(term, ratMul(c[k], xv[k]));
            MLK_TRY_VAR(added, ratAdd(acc, term));
            acc = added;
        }
        outObj = acc;
        outX = std::move(xv);
        return LpStatus::Optimal;
    }

private:
    [[nodiscard]] bool fillStructural(const SmallVector<Rational, 8>& src,
                                      SmallVector<Rational, 16>& row) {
        for (uint32_t k = 0; k * 2 < nStruct_ && k < src.size(); ++k) {
            const Rational& c = src[k];
            if (c.isZero()) continue;
            row[2 * k] = c;
            Result<Rational> neg = ratNeg(c);
            if (!neg.has_value()) return false;
            row[2 * k + 1] = *neg;
        }
        return true;
    }

    [[nodiscard]] static bool negateRow(SmallVector<Rational, 16>& row) {
        for (Rational& v : row) {
            Result<Rational> n = ratNeg(v);
            if (!n.has_value()) return false;
            v = *n;
        }
        return true;
    }

    [[nodiscard]] static bool negateInto(const Rational& in, Rational& out) {
        Result<Rational> n = ratNeg(in);
        if (!n.has_value()) return false;
        out = *n;
        return true;
    }

    /// Runs Bland-rule iterations on cost `cost`; entering columns are
    /// restricted to [0, maxEnter). The reduced-cost row starts as the
    /// cost vector and is canonicalized against the current basis
    /// (standard c̄_j = c_j - c_B B^-1 A_j).
    [[nodiscard]] Result<LpStatus> runSimplex(
        const SmallVector<Rational, 16>& cost, uint32_t maxEnter) {
        SmallVector<Rational, 16> z(nCols_ + 1, Rational{});
        for (uint32_t col = 0; col < nCols_; ++col) {
            z[col] = cost[col];
        }
        for (uint32_t r = 0; r < m_; ++r) {
            const uint32_t bcol = basis_[r];
            const Rational zc = z[bcol];
            if (zc.isZero()) continue;
            for (uint32_t col = 0; col < nCols_; ++col) {
                MLK_TRY_VAR(t, ratMul(zc, rows_[r][col]));
                MLK_TRY_VAR(s, ratSub(z[col], t));
                z[col] = s;
            }
            MLK_TRY_VAR(t2, ratMul(zc, rhs_[r]));
            MLK_TRY_VAR(s2, ratSub(z[nCols_], t2));
            z[nCols_] = s2;
        }
        for (uint64_t iter = 0; iter < static_cast<uint64_t>(kLpMaxVars) * 4;
             ++iter) {
            uint32_t enter = nCols_;
            for (uint32_t col = 0; col < maxEnter && col < nCols_; ++col) {
                if (z[col].sign() < 0) {
                    enter = col;  // Bland: lowest index
                    break;
                }
            }
            if (enter == nCols_) return LpStatus::Optimal;
            uint32_t leave = m_;
            Rational bestRatio{};
            for (uint32_t r = 0; r < m_; ++r) {
                const Rational& piv = rows_[r][enter];
                if (piv.sign() <= 0) continue;
                MLK_TRY_VAR(ratio, ratDiv(rhs_[r], piv));
                if (leave == m_ || ratLess(ratio, bestRatio)) {
                    leave = r;
                    bestRatio = ratio;
                } else if (!ratLess(bestRatio, ratio) &&
                           basis_[r] < basis_[leave]) {
                    leave = r;  // Bland tie-break: lowest basis column
                }
            }
            if (leave == m_) return LpStatus::Unbounded;
            MLK_TRYV(pivot(leave, enter));
            // Push the pivot through the objective row.
            const Rational zc = z[enter];
            if (!zc.isZero()) {
                for (uint32_t col = 0; col < nCols_; ++col) {
                    MLK_TRY_VAR(t, ratMul(zc, rows_[leave][col]));
                    MLK_TRY_VAR(s, ratSub(z[col], t));
                    z[col] = s;
                }
                MLK_TRY_VAR(t2, ratMul(zc, rhs_[leave]));
                MLK_TRY_VAR(s2, ratSub(z[nCols_], t2));
                z[nCols_] = s2;
            }
        }
        return err(ErrorCode::BudgetExceeded, "lp iteration budget");
    }

    [[nodiscard]] Result<void> pivot(uint32_t r, uint32_t c) {
        const Rational p = rows_[r][c];
        if (p.isZero()) {
            return err(ErrorCode::Internal, "lp pivot on zero element");
        }
        for (uint32_t col = 0; col < nCols_; ++col) {
            MLK_TRY_VAR(q, ratDiv(rows_[r][col], p));
            rows_[r][col] = q;
        }
        MLK_TRY_VAR(b, ratDiv(rhs_[r], p));
        rhs_[r] = b;
        for (uint32_t rr = 0; rr < m_; ++rr) {
            if (rr == r) continue;
            const Rational f = rows_[rr][c];
            if (f.isZero()) continue;
            for (uint32_t col = 0; col < nCols_; ++col) {
                MLK_TRY_VAR(t, ratMul(f, rows_[r][col]));
                MLK_TRY_VAR(s, ratSub(rows_[rr][col], t));
                rows_[rr][col] = s;
            }
            MLK_TRY_VAR(t2, ratMul(f, rhs_[r]));
            MLK_TRY_VAR(s2, ratSub(rhs_[rr], t2));
            rhs_[rr] = s2;
        }
        basis_[r] = c;
        return {};
    }

    [[nodiscard]] Result<SmallVector<Rational, 8>> extractX() const {
        SmallVector<Rational, 8> x(nVars_, Rational{});
        for (uint32_t r = 0; r < m_; ++r) {
            const uint32_t b = basis_[r];
            if (b >= nStruct_) continue;
            const uint32_t var = b / 2;
            if (var >= x.size()) continue;
            if (b % 2 == 0) {
                x[var] = rhs_[r];
            } else {
                MLK_TRY_VAR(nv, ratNeg(rhs_[r]));
                x[var] = nv;
            }
        }
        return x;
    }

    uint32_t nVars_{0};
    uint32_t nStruct_{0};
    uint32_t nSurplus_{0};
    uint32_t nArt_{0};
    uint32_t nCols_{0};
    uint32_t m_{0};
    SmallVector<SmallVector<Rational, 16>, 16> rows_{};
    SmallVector<Rational, 16> rhs_{};
    SmallVector<uint32_t, 16> basis_{};
};

}  // namespace

Result<LpSolution> solveLp(const LinearProgram& lp) {
    if (lp.nVars == 0) {
        return err(ErrorCode::InvalidArgument, "lp with zero variables");
    }
    Simplex sx;
    if (!sx.build(lp)) {
        return err(ErrorCode::ResourceExhausted,
                   "lp exceeds solver size budgets");
    }
    MLK_TRY_VAR(feasible, sx.phase1());
    LpSolution out;
    if (!feasible) {
        out.status = LpStatus::Infeasible;
        return out;
    }
    Rational obj{};
    SmallVector<Rational, 8> x;
    MLK_TRY_VAR(st, sx.phase2(lp.objective, obj, x));
    out.status = st;
    out.objective = obj;
    out.x = std::move(x);
    return out;
}

}  // namespace mlk::poly
