// Backend passes (spec §8.12): backend.emit_binary. The CPU backend
// artifact = serialized KernelModule (stable JSON, Rule 24) + C++ source
// emission for AOT packages. No machine code is generated in-process
// (kernel_abi.md: publication is via atomically swapped realization
// pointers; W^X rules apply to executable pages, of which MLK+ MVP has
// none — documented in docs/kernel_abi.md).
#include "../passes_common.h"

namespace mlk::passes {

class EmitBinaryPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        if (ctx.kernelOut == nullptr) {
            return err(ErrorCode::InvalidArgument,
                       "backend.emit_binary requires kernelOut sink");
        }
        KernelModule& kernel = *ctx.kernelOut;
        if (kernel.nodes.empty()) {
            return err(ErrorCode::InvalidGraph,
                       "emit_binary: empty kernel module (run "
                       "lower.to_kernel_ir first)");
        }
        // Fallback metadata is a REQUIRED compilation output (Rule 100):
        // every kernel carries the Tier-0 re-entry information.
        const SymbolId fallbackAttr = ctx.symbols->intern("fallback_tier0");
        if (kernel.scheduleParams.find(fallbackAttr) == nullptr) {
            (void)kernel.scheduleParams.findOrInsert(fallbackAttr, nullptr, 1);
        }
        (void)graph;  // math graph untouched (Rule 23)
        r.changed = false;
        return r;
    }
};

void registerBackendPasses(SymbolTable& symbols) {
    static EmitBinaryPass emit(symbols, "backend.emit_binary",
                               PassKind::Lowering);
    registerPass(symbols, emit, PassKind::Lowering, {"kernel.built"},
                 {"kernel.emitted"}, {},
                 {Tier::Tier1, Tier::Tier2, Tier::Tier3});
}

}  // namespace mlk::passes
