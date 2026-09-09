// MLK+ intervals (property/interval.h): range facts attached to values.
#pragma once

#include <cmath>
#include <limits>

namespace mlk {

/// Closed/open interval over the reals; +-inf allowed via infinity().
struct Interval {
    double low{0.0};
    double high{0.0};
    bool inclusiveLow{true};
    bool inclusiveHigh{true};

    [[nodiscard]] static Interval everything() {
        return Interval{-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(), false,
                        false};
    }
    [[nodiscard]] static Interval point(double v) { return Interval{v, v}; }
    [[nodiscard]] static Interval closed(double lo, double hi) {
        return Interval{lo, hi, true, true};
    }
    [[nodiscard]] static Interval unit() { return closed(0.0, 1.0); }
    [[nodiscard]] static Interval nonNegative() {
        return Interval{0.0, std::numeric_limits<double>::infinity(), true,
                        false};
    }

    [[nodiscard]] bool contains(double v) const {
        const bool geLow = v > low || (inclusiveLow && v >= low) || v == low;
        const bool leHigh = v < high || (inclusiveHigh && v <= high) ||
                            v == high;
        return geLow && leHigh;
    }

    [[nodiscard]] bool isEmpty() const {
        return low > high || (low == high && !(inclusiveLow && inclusiveHigh));
    }

    [[nodiscard]] bool isPoint() const {
        return low == high && inclusiveLow && inclusiveHigh;
    }

    /// Hull union (used to merge range facts).
    [[nodiscard]] static Interval hull(const Interval& a, const Interval& b) {
        Interval r;
        if (a.low < b.low) {
            r.low = a.low;
            r.inclusiveLow = a.inclusiveLow;
        } else if (a.low > b.low) {
            r.low = b.low;
            r.inclusiveLow = b.inclusiveLow;
        } else {
            r.low = a.low;
            r.inclusiveLow = a.inclusiveLow || b.inclusiveLow;
        }
        if (a.high > b.high) {
            r.high = a.high;
            r.inclusiveHigh = a.inclusiveHigh;
        } else if (a.high < b.high) {
            r.high = b.high;
            r.inclusiveHigh = b.inclusiveHigh;
        } else {
            r.high = a.high;
            r.inclusiveHigh = a.inclusiveHigh || b.inclusiveHigh;
        }
        return r;
    }

    [[nodiscard]] bool operator==(const Interval& o) const {
        return low == o.low && high == o.high &&
               inclusiveLow == o.inclusiveLow && inclusiveHigh == o.inclusiveHigh;
    }
};

}  // namespace mlk
