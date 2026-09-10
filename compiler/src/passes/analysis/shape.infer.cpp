// shape.infer — shape/dimensionality inference pass (spec §8.1).
#include "../passes_common.h"
#include "mlk/type/type_inference.h"

namespace mlk::passes {

namespace {
constexpr Tier kAllTiers[] = {Tier::Tier0, Tier::Tier1, Tier::Tier2,
                              Tier::Tier3};
}  // namespace

class ShapeInferPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        MLK_TRY_VAR(validated, inferShapes(graph));
        (void)validated;
        return r;
    }
};

void register_shape_infer_pass(SymbolTable& symbols) {
    static ShapeInferPass pass(symbols, "shape.infer", PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"type.inferred"},
                 {"shape.inferred"}, {},
                 {kAllTiers[0], kAllTiers[1], kAllTiers[2], kAllTiers[3]});
}

}  // namespace mlk::passes
