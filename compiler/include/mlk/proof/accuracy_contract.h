// MLK+ AccuracyContract (Rule 34: no approximation without an error
// contract; Rule 91: approximation is a semantic change).
#pragma once

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/json.h"

namespace mlk {

struct AccuracyContract {
    double maxUlps{constants::kDefaultMaxUlps};
    double maxAbsError{constants::kDefaultMaxAbsError};
    double maxRelError{constants::kDefaultMaxRelError};
    bool allowFastMath{false};
    bool allowApproximateReciprocal{false};
    bool allowApproximateDivide{false};
    bool preserveNaN{true};
    bool preserveInf{true};
    bool allowReassociation{false};

    [[nodiscard]] bool permitsApproximation() const {
        // Rule 34: approximations are forbidden when the contract is absent
        // or violated. A contract exists when any approximate allowance is
        // explicitly requested AND error bounds are finite.
        return (allowFastMath || allowApproximateReciprocal ||
                allowApproximateDivide || allowReassociation) &&
               maxUlps > 0.0;
    }

    [[nodiscard]] bool operator==(const AccuracyContract& o) const {
        return maxUlps == o.maxUlps && maxAbsError == o.maxAbsError &&
               maxRelError == o.maxRelError &&
               allowFastMath == o.allowFastMath &&
               allowApproximateReciprocal == o.allowApproximateReciprocal &&
               allowApproximateDivide == o.allowApproximateDivide &&
               preserveNaN == o.preserveNaN && preserveInf == o.preserveInf &&
               allowReassociation == o.allowReassociation;
    }

    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashF64(maxUlps);
        h = hashCombine(h, hashF64(maxAbsError));
        h = hashCombine(h, hashF64(maxRelError));
        h = hashCombine(h, hashU64(allowFastMath));
        h = hashCombine(h, hashU64(allowReassociation));
        return h;
    }

    [[nodiscard]] json::Value toJson() const;
};

[[nodiscard]] Result<AccuracyContract> accuracyContractFromJson(
    const json::Value& doc);

}  // namespace mlk
