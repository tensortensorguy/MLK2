// property.infer — tri-state mathematical property inference (spec §8.1;
// associative/commutative/distributive are three-state facts derived from
// operand domains, never booleans).
#include "../passes_common.h"
#include "mlk/property/property_inference.h"

namespace mlk::passes {

namespace {
constexpr Tier kAllTiers[] = {Tier::Tier0, Tier::Tier1, Tier::Tier2,
                              Tier::Tier3};
}  // namespace

class PropertyInferPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        MLK_TRY_VAR(established, inferProperties(graph, *ctx.domainProfile));
        r.changed = established != 0;
        r.invalidatedAnalyses.push_back(ctx.symbols->intern("analysis.cse"));
        return r;
    }
};

void register_property_infer_pass(SymbolTable& symbols) {
    static PropertyInferPass pass(symbols, "property.infer",
                                  PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"type.inferred"},
                 {"property.inferred"}, {"analysis.cse"},
                 {kAllTiers[0], kAllTiers[1], kAllTiers[2], kAllTiers[3]});
}

}  // namespace mlk::passes
