// MLK+ safepoints (Rule 107/130: bounded-latency polls in long-running
// kernels; cancellation cooperative).
#pragma once

#include "mlk/core/cancellation.h"

namespace mlk {

/// Safepoint poll: called every kSafepointPollIntervalIterations in
/// generated/interpreted long loops. Latency is bounded by construction
/// (single atomic load).
class Safepoint {
public:
    explicit Safepoint(CancellationToken& token) : token_(token) {}

    void poll(std::uint64_t iteration) {
        if (iteration % constants::kSafepointPollIntervalIterations == 0) {
            if (token_.cancelled()) cancelled_ = true;
        }
    }

    [[nodiscard]] bool cancelled() const noexcept { return cancelled_; }

private:
    CancellationToken& token_;
    bool cancelled_{false};
};

}  // namespace mlk
