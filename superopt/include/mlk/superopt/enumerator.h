// MLK+ scalar straight-line superoptimizer enumerator (spec §6.2):
// searches small instruction sequences — FMA formation, square forms,
// strength reduction. Every candidate carries a certificate (Rule 52) and
// is verified before benchmarking (Rule 50).
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/proof/accuracy_contract.h"
#include "mlk/superopt/superoptimizer.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

/// Rewrites x*x + 3x-style scalar chains into FMA-friendly algebraic forms.
class ScalarPeepholeSuperoptimizer final : public Superoptimizer {
public:
    explicit ScalarPeepholeSuperoptimizer(SymbolTable& symbols);
    [[nodiscard]] SymbolId name() const override { return name_; }
    [[nodiscard]] bool canApply(const MathGraph& graph,
                                const MathDomainProfile& profile) override;
    [[nodiscard]] Result<SmallVector<RealizationCandidate, 8>> generate(
        const MathGraph& graph, const AccuracyContract& contract,
        const MathDomainProfile& profile) override;

private:
    SymbolTable& symbols_;
    SymbolId name_;
};

/// Math-function approximation superoptimizer (spec §6.4): polynomial
/// families with range reduction; emits candidates with measured ULP error
/// bounds (Rule 34/50: verified before use).
class MathFunctionApproximator final : public Superoptimizer {
public:
    explicit MathFunctionApproximator(SymbolTable& symbols);
    [[nodiscard]] SymbolId name() const override { return name_; }
    [[nodiscard]] bool canApply(const MathGraph& graph,
                                const MathDomainProfile& profile) override;
    [[nodiscard]] Result<SmallVector<RealizationCandidate, 8>> generate(
        const MathGraph& graph, const AccuracyContract& contract,
        const MathDomainProfile& profile) override;

private:
    SymbolTable& symbols_;
    SymbolId name_;
};

/// Algebraic e-graph optimizer (spec §6.1): saturation + multi-extract.
class AlgebraicEGraphOptimizer final : public Superoptimizer {
public:
    explicit AlgebraicEGraphOptimizer(SymbolTable& symbols);
    [[nodiscard]] SymbolId name() const override { return name_; }
    [[nodiscard]] bool canApply(const MathGraph& graph,
                                const MathDomainProfile& profile) override;
    [[nodiscard]] Result<SmallVector<RealizationCandidate, 8>> generate(
        const MathGraph& graph, const AccuracyContract& contract,
        const MathDomainProfile& profile) override;

private:
    SymbolTable& symbols_;
    SymbolId name_;
};

}  // namespace mlk
