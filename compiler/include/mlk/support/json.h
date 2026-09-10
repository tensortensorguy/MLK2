// MLK+ minimal hermetic JSON (layout spec: schemas/ + profiles/ + .mlk graph
// files are JSON; Rule 160: hermetic builds, pinned dependencies).
//
// This is a small, allocation-bounded DOM parser/serializer sufficient for
// .mlk graph files, domain profiles, realization cache entries, and
// telemetry. It uses no exceptions: parse errors return Result.
// Object key order is preserved (determinism, Rule 143).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "mlk/core/result.h"

namespace mlk::json {

class Value;
using Array = std::vector<Value>;
/// Object member: ordered key/value pair. Value is incomplete here, so the
/// member value is heap-indirected (shared_ptr type-erases the deleter).
struct Member {
    std::string key;
    std::shared_ptr<Value> value;
};
using Object = std::vector<Member>;

/// Kind of a JSON value. Named Kind (not Type) to avoid confusion with the
/// value-class names below.
/// GCC -Wshadow quirk (PR 90467 family): enum class enumerators are scoped
/// to the enum, yet GCC still reports them shadowing the same-scope aliases
/// `Array`/`Object`. The shadow is unreachable (lookup requires Kind::),
/// so suppress exactly here, with both names kept: the aliases are the
/// public container API, the enumerators the kind tags.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
enum class Kind : uint8_t { Null, Bool, Int, Double, String, Array, Object };
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

class Value {
public:
    Value() : storage_(std::nullptr_t{}) {}
    Value(std::nullptr_t) : storage_(std::nullptr_t{}) {}
    Value(bool b) : storage_(b) {}
    Value(int64_t i) : storage_(i) {}
    Value(double d) : storage_(d) {}
    Value(std::string s) : storage_(std::move(s)) {}
    Value(const char* s) : storage_(std::string(s)) {}
    Value(Array a) : storage_(std::move(a)) {}
    Value(Object o) : storage_(std::move(o)) {}

    [[nodiscard]] Kind type() const {
        if (std::holds_alternative<std::nullptr_t>(storage_)) return Kind::Null;
        if (std::holds_alternative<bool>(storage_)) return Kind::Bool;
        if (std::holds_alternative<int64_t>(storage_)) return Kind::Int;
        if (std::holds_alternative<double>(storage_)) return Kind::Double;
        if (std::holds_alternative<std::string>(storage_)) return Kind::String;
        if (std::holds_alternative<Array>(storage_)) return Kind::Array;
        return Kind::Object;
    }

    [[nodiscard]] bool isNull() const { return type() == Kind::Null; }
    [[nodiscard]] bool isBool() const { return type() == Kind::Bool; }
    [[nodiscard]] bool isInt() const { return type() == Kind::Int; }
    [[nodiscard]] bool isDouble() const { return type() == Kind::Double; }
    [[nodiscard]] bool isNumber() const { return isInt() || isDouble(); }
    [[nodiscard]] bool isString() const { return type() == Kind::String; }
    [[nodiscard]] bool isArray() const { return type() == Kind::Array; }
    [[nodiscard]] bool isObject() const { return type() == Kind::Object; }

    [[nodiscard]] bool asBool() const { return std::get<bool>(storage_); }
    [[nodiscard]] bool isObjectOrArray() const {
        return isObject() || isArray();
    }
    [[nodiscard]] int64_t asInt() const {
        if (std::holds_alternative<int64_t>(storage_)) {
            return std::get<int64_t>(storage_);
        }
        return static_cast<int64_t>(std::get<double>(storage_));
    }
    [[nodiscard]] double asDouble() const {
        if (std::holds_alternative<double>(storage_)) {
            return std::get<double>(storage_);
        }
        return static_cast<double>(std::get<int64_t>(storage_));
    }
    [[nodiscard]] const std::string& asString() const {
        return std::get<std::string>(storage_);
    }
    [[nodiscard]] const Array& asArray() const {
        return std::get<Array>(storage_);
    }
    [[nodiscard]] Array& asArray() { return std::get<Array>(storage_); }
    [[nodiscard]] const Object& asObject() const {
        return std::get<Object>(storage_);
    }
    [[nodiscard]] Object& asObject() { return std::get<Object>(storage_); }

    // Object accessors ------------------------------------------------------
    [[nodiscard]] const Value* find(std::string_view key) const {
        if (!isObject()) return nullptr;
        for (const auto& m : asObject()) {
            if (m.key == key) return m.value.get();
        }
        return nullptr;
    }

    void set(std::string key, Value v) {
        if (!isObject()) {
            storage_ = Object{};
        }
        auto& obj = asObject();
        for (auto& m : obj) {
            if (m.key == key) {
                *m.value = std::move(v);
                return;
            }
        }
        obj.push_back(
            {std::move(key),
             std::shared_ptr<Value>(new Value(std::move(v)))});
    }

    /// Appends a null-initialized member; returns reference to the value for
    /// in-place construction.
    [[nodiscard]] Value& emplaceMember(std::string key) {
        if (!isObject()) storage_ = Object{};
        asObject().push_back(
            {std::move(key), std::shared_ptr<Value>(new Value())});
        return *asObject().back().value;
    }

    void push(Value v) {
        if (!isArray()) storage_ = Array{};
        asArray().push_back(std::move(v));
    }

private:
    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array,
                 Object>
        storage_;
};

/// Parses `text` into a DOM. Rule 124: malformed input is rejected safely.
[[nodiscard]] Result<Value> parse(std::string_view text);

/// Serializes to compact JSON. Doubles use shortest round-trip formatting.
[[nodiscard]] std::string serialize(const Value& v);

/// Serializes with 2-space indentation.
[[nodiscard]] std::string serializePretty(const Value& v);

}  // namespace mlk::json
