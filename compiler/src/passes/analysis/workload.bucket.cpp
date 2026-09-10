// workload.bucket — stable dynamic-dimension bucketing for the autotuner
// (spec §8.1; Rule 23: bucket ids ride in fact payloads, not IR).
#include "../passes_common.h"

namespace mlk::passes {

class WorkloadBucketPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        // Bucket dynamic dims into stable tuning buckets
        // [1..32][33..128][129..512][513..2048]+.
        for (const ValueId vid : graph.outputs()) {
            const Value& v = graph.value(vid);
            int64_t bucket = 0;
            if (v.type.tensor) {
                const Shape& s = v.type.tensor->shape;
                for (std::size_t i = 0; i < s.rank(); ++i) {
                    const int64_t d = s.dim(i);
                    if (d == kDynamicDim) continue;
                    int64_t b = 0;
                    for (std::size_t e = 0;
                         e < constants::kWorkloadBucketEdges.size(); ++e) {
                        if (d <= constants::kWorkloadBucketEdges[e]) {
                            b = static_cast<int64_t>(e) + 1;
                            break;
                        }
                    }
                    if (b > bucket) bucket = b;
                }
            }
            // The bucket is a fact about the workload, not the math: it is
            // recorded in the value's fact payload (period field reused as
            // the bucket id) — Rule 23 (no IR contamination beyond facts).
            Fact f;
            f.property = PropertyId::Contiguous;
            f.value = TriState::True;
            f.period = static_cast<double>(bucket);
            graph.value(vid).facts.set(f);
        }
        r.changed = false;
        return r;
    }
};

void register_workload_bucket_pass(SymbolTable& symbols) {
    static WorkloadBucketPass pass(symbols, "workload.bucket",
                                   PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"shape.inferred"},
                 {"workload.bucketed"}, {},
                 {Tier::Tier1, Tier::Tier2, Tier::Tier3});
}

}  // namespace mlk::passes
