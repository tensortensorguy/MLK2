// AccuracyContract JSON (de)serialization (versioned; Rule 37).
#include "mlk/proof/accuracy_contract.h"

namespace mlk {

json::Value AccuracyContract::toJson() const {
    json::Value v = json::Object{};
    v.set("max_ulps", json::Value{maxUlps});
    v.set("max_abs_error", json::Value{maxAbsError});
    v.set("max_rel_error", json::Value{maxRelError});
    v.set("allow_fast_math", json::Value{allowFastMath});
    v.set("allow_approximate_reciprocal",
          json::Value{allowApproximateReciprocal});
    v.set("allow_approximate_divide", json::Value{allowApproximateDivide});
    v.set("preserve_nan", json::Value{preserveNaN});
    v.set("preserve_inf", json::Value{preserveInf});
    v.set("allow_reassociation", json::Value{allowReassociation});
    return v;
}

Result<AccuracyContract> accuracyContractFromJson(const json::Value& doc) {
    AccuracyContract c;
    auto getF = [&](const char* k, double& out) -> Result<Ok> {
        if (const json::Value* v = doc.find(k)) {
            if (!v->isNumber() || v->asDouble() < 0.0) {
                return err(ErrorCode::ParseError,
                           std::string(k) + " must be a non-negative number");
            }
            out = v->asDouble();
        }
        return ok();
    };
    auto getB = [&](const char* k, bool& out) -> Result<Ok> {
        if (const json::Value* v = doc.find(k)) {
            if (!v->isBool()) {
                return err(ErrorCode::ParseError,
                           std::string(k) + " must be a bool");
            }
            out = v->asBool();
        }
        return ok();
    };
    MLK_TRYV(getF("max_ulps", c.maxUlps));
    MLK_TRYV(getF("max_abs_error", c.maxAbsError));
    MLK_TRYV(getF("max_rel_error", c.maxRelError));
    MLK_TRYV(getB("allow_fast_math", c.allowFastMath));
    MLK_TRYV(getB("allow_approximate_reciprocal", c.allowApproximateReciprocal));
    MLK_TRYV(getB("allow_approximate_divide", c.allowApproximateDivide));
    MLK_TRYV(getB("preserve_nan", c.preserveNaN));
    MLK_TRYV(getB("preserve_inf", c.preserveInf));
    MLK_TRYV(getB("allow_reassociation", c.allowReassociation));
    return c;
}

}  // namespace mlk
