// Integer affine expression implementation (see affine_expr.h).
#include "mlk/poly/affine_expr.h"

#include <string>

#include "mlk/poly/checked.h"

namespace mlk::poly {

AffineExpr AffineExpr::fromConstant(const VarSpace& space, int64_t v) noexcept {
    AffineExpr e;
    e.space = space;
    e.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
    e.constant = v;
    return e;
}

AffineExpr AffineExpr::variable(const VarSpace& space,
                                uint32_t varIndex) noexcept {
    AffineExpr e;
    e.space = space;
    e.coeffs = SmallVector<int64_t, 8>(space.totalVars(), 0);
    if (varIndex < e.coeffs.size()) {
        e.coeffs[varIndex] = 1;
    }
    return e;
}

bool AffineExpr::isConstant(int64_t* value) const noexcept {
    for (const int64_t c : coeffs) {
        if (c != 0) return false;
    }
    if (value != nullptr) *value = constant;
    return true;
}

int64_t AffineExpr::coeffOf(uint32_t var) const noexcept {
    return var < coeffs.size() ? coeffs[var] : 0;
}

bool AffineExpr::withinLimits() const noexcept {
    if (constant > kRationalMagnitudeLimit ||
        constant < -kRationalMagnitudeLimit) {
        return false;
    }
    for (const int64_t c : coeffs) {
        if (c > kRationalMagnitudeLimit || c < -kRationalMagnitudeLimit) {
            return false;
        }
    }
    return true;
}

void AffineExpr::gcdNormalize() noexcept {
    int64_t g = 0;
    for (const int64_t c : coeffs) {
        g = ratGcd(g, c);
        if (g == 1) return;  // already minimal; keep constant untouched
    }
    g = ratGcd(g, constant);
    if (g <= 1) return;
    for (int64_t& c : coeffs) {
        c /= g;
    }
    constant /= g;
}

Result<int64_t> AffineExpr::evalAt(
    const SmallVector<int64_t, 8>& vars) const noexcept {
    if (vars.size() < coeffs.size()) {
        return err(ErrorCode::InvalidArgument,
                   "affine eval: variable vector shorter than space");
    }
    int64_t acc = constant;
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i] == 0) continue;
        int64_t term = 0;
        if (!checked::mul(coeffs[i], vars[i], &term) ||
            !checked::add(acc, term, &acc)) {
            return err(ErrorCode::ResourceExhausted,
                       "affine eval overflows int64");
        }
    }
    return acc;
}

bool exprSpacesMatch(const AffineExpr& a, const AffineExpr& b) noexcept {
    return a.space.sameAs(b.space);
}

bool exprEqual(const AffineExpr& a, const AffineExpr& b) noexcept {
    if (!exprSpacesMatch(a, b)) return false;
    if (a.constant != b.constant) return false;
    if (a.coeffs.size() != b.coeffs.size()) return false;
    for (std::size_t i = 0; i < a.coeffs.size(); ++i) {
        if (a.coeffs[i] != b.coeffs[i]) return false;
    }
    return true;
}

Result<AffineExpr> exprAdd(const AffineExpr& a, const AffineExpr& b) noexcept {
    if (!exprSpacesMatch(a, b)) {
        return err(ErrorCode::InvalidArgument,
                   "affine add: variable spaces differ");
    }
    AffineExpr out;
    out.space = a.space;
    MLK_TRY_VAR(cst, checked::addLimited(a.constant, b.constant));
    out.constant = cst;
    out.coeffs.reserve(a.coeffs.size());
    for (std::size_t i = 0; i < a.coeffs.size(); ++i) {
        MLK_TRY_VAR(s, checked::addLimited(a.coeffs[i], b.coeffs[i]));
        out.coeffs.push_back(s);
    }
    return out;
}

Result<AffineExpr> exprSub(const AffineExpr& a, const AffineExpr& b) noexcept {
    MLK_TRY_VAR(negb, exprScale(b, -1));
    return exprAdd(a, negb);
}

Result<AffineExpr> exprScale(const AffineExpr& a, int64_t k) noexcept {
    AffineExpr out;
    out.space = a.space;
    MLK_TRY_VAR(cst, checked::mulLimited(a.constant, k));
    out.constant = cst;
    out.coeffs.reserve(a.coeffs.size());
    for (const int64_t c : a.coeffs) {
        MLK_TRY_VAR(s, checked::mulLimited(c, k));
        out.coeffs.push_back(s);
    }
    return out;
}

std::string exprToString(const AffineExpr& e,
                         const SmallVector<std::string, 8>& varNames) {
    // Cold path: tooling/diagnostics only (Rule 16: no strings in IR).
    std::string out;
    bool first = true;
    auto appendTerm = [&](int64_t c, const std::string& name) {
        if (c == 0) return;
        if (!first) out += (c < 0 ? " - " : " + ");
        else if (c < 0) out += "-";
        const int64_t mag = c < 0 ? -c : c;
        if (!(mag == 1 && !name.empty())) {
            out += std::to_string(mag);
            if (!name.empty()) out += "*";
        }
        out += name;
        first = false;
    };
    for (std::size_t i = 0; i < e.coeffs.size(); ++i) {
        const std::string name =
            i < varNames.size() ? varNames[i] : ("x" + std::to_string(i));
        appendTerm(e.coeffs[i], name);
    }
    if (e.constant != 0 || first) {
        if (!first) {
            out += e.constant < 0 ? " - " : " + ";
            out += std::to_string(e.constant < 0 ? -e.constant : e.constant);
        } else {
            out += std::to_string(e.constant);
        }
    }
    return out;
}

}  // namespace mlk::poly
