// poly.tile — detect the maximal tileable band prefix and record tile
// sizes (spec §8.13; docs/polyhedral_spec.md §tiling). Transform pass:
// the loop rewriting happens in poly.codegen using this record.
//
// Contract (Rule 142):
//   required  : poly.scop, poly.deps, poly.schedule
//   produced  : poly.tile
//   invalidated: poly.tile, poly.codegen
//   kill switch: "poly.tile" (Rule 60)
#include "../passes_common.h"
#include "mlk/poly/tile.h"
#include "mlk/poly/workspace.h"

namespace mlk::passes {

namespace {
constexpr Tier kPolyTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class PolyTilePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        if (ctx.killed(nameId(*ctx.symbols))) return r;  // Rule 60
        if (ctx.kernelOut == nullptr || ctx.polyWorkspace == nullptr) {
            return r;
        }
        poly::PolyWorkspace& ws = *ctx.polyWorkspace;
        if (!ws.scopValid || !ws.dependencesValid || !ws.scheduleValid) {
            return r;
        }
        ws.tileValid = false;
        ws.codegenValid = false;

        // Knob override (Rule 29): poly_tile_size, default from constants.
        int64_t tileSize = constants::kPolyDefaultTileSize;
        if (ctx.knobs != nullptr) {
            if (const int64_t* v = ctx.knobs->find(
                    ctx.symbols->intern("poly_tile_size"))) {
                if (*v > 0) tileSize = *v;
            }
        }
        MLK_TRY_VAR(tiled,
                    poly::computeTiling(ws.scop, ws.dependences,
                                        ws.schedule, tileSize));
        ws.statsTiledBands = tiled.tiled ? 1 : 0;
        ws.tiled = std::move(tiled);
        ws.tileValid = true;
        return r;
    }
};

void register_poly_tile_pass(SymbolTable& symbols) {
    static PolyTilePass pass(symbols, "poly.tile", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform,
                 {"poly.scop", "poly.deps", "poly.schedule"},
                 {"poly.tile"}, {"poly.tile", "poly.codegen"},
                 {kPolyTiers[0], kPolyTiers[1]});
}

}  // namespace mlk::passes
