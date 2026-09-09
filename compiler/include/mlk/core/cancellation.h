// MLK+ cancellation token (Rule 53: compile=INF must be cancellable;
// Rule 130/132: bounded latency, cancellable compilation/tuning).
#pragma once

#include <atomic>

namespace mlk {

class CancellationToken {
public:
    void cancel() noexcept { flag_.store(true, std::memory_order_release); }
    [[nodiscard]] bool cancelled() const noexcept {
        return flag_.load(std::memory_order_acquire);
    }
    void reset() noexcept { flag_.store(false, std::memory_order_relaxed); }

private:
    std::atomic<bool> flag_{false};
};

}  // namespace mlk
