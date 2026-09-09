// MLK+ facts: first-class property records attached to values and nodes
// (Rule 22). A fact is either tri-state or carries a payload (period,
// interval). FactSets are small, inline, deterministic.
#pragma once

#include <optional>

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/core/small_vector.h"
#include "mlk/property/interval.h"
#include "mlk/property/property_lattice.h"

namespace mlk {

/// A single established fact about a value or node.
struct Fact {
    PropertyId property{PropertyId::Commutative};
    TriState value{TriState::Unknown};
    /// Payload for Periodic (period in the input's units).
    double period{0.0};
    /// Payload for Bounded / range facts.
    Interval range{Interval::everything()};

    [[nodiscard]] bool operator==(const Fact& o) const {
        return property == o.property && value == o.value &&
               period == o.period && range == o.range;
    }
};

/// Ordered set of facts (deterministic; Rule 143).
class FactSet {
public:
    /// Returns previous fact pointer if the property already existed.
    void set(Fact f) {
        for (auto& existing : facts_) {
            if (existing.property == f.property) {
                existing = f;
                return;
            }
        }
        facts_.push_back(f);
    }

    void setTriState(PropertyId p, TriState v) {
        Fact f;
        f.property = p;
        f.value = v;
        set(f);
    }

    void setRange(Interval r) {
        Fact f;
        f.property = PropertyId::Bounded;
        f.value = TriState::True;
        f.range = r;
        set(f);
    }

    void setPeriod(double p) {
        Fact f;
        f.property = PropertyId::Periodic;
        f.value = TriState::True;
        f.period = p;
        set(f);
    }

    [[nodiscard]] const Fact* get(PropertyId p) const {
        for (const auto& f : facts_) {
            if (f.property == p) return &f;
        }
        return nullptr;
    }

    [[nodiscard]] TriState triState(PropertyId p) const {
        const Fact* f = get(p);
        return f ? f->value : TriState::Unknown;
    }

    /// Rule 22 gate: only a definite True counts.
    [[nodiscard]] bool isTrue(PropertyId p) const {
        return triState(p) == TriState::True;
    }
    [[nodiscard]] bool isFalse(PropertyId p) const {
        return triState(p) == TriState::False;
    }

    [[nodiscard]] const Interval* range() const {
        const Fact* f = get(PropertyId::Bounded);
        return f ? &f->range : nullptr;
    }

    void clear() { facts_.clear(); }
    [[nodiscard]] bool empty() const { return facts_.empty(); }
    [[nodiscard]] std::size_t size() const { return facts_.size(); }
    [[nodiscard]] const SmallVector<Fact, constants::kFactsInline>& all()
        const {
        return facts_;
    }

    [[nodiscard]] bool operator==(const FactSet& o) const {
        if (facts_.size() != o.facts_.size()) return false;
        for (std::size_t i = 0; i < facts_.size(); ++i) {
            if (!(facts_[i] == o.facts_[i])) return false;
        }
        return true;
    }

    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = kHashSeed;
        for (const auto& f : facts_) {
            h = hashCombine(h, hashU64(static_cast<uint64_t>(f.property)));
            h = hashCombine(h, hashU64(static_cast<uint64_t>(f.value)));
            if (f.property == PropertyId::Bounded) {
                h = hashCombine(h, hashF64(f.range.low));
                h = hashCombine(h, hashF64(f.range.high));
            }
            if (f.property == PropertyId::Periodic) {
                h = hashCombine(h, hashF64(f.period));
            }
        }
        return h;
    }

private:
    SmallVector<Fact, constants::kFactsInline> facts_{};
};

}  // namespace mlk
