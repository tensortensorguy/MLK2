// Minimal hermetic JSON parser/serializer implementation.
// See mlk/support/json.h for the contract. Rule 124: malformed input is
// rejected safely (no UB, no over-long allocation via nested-depth bound).
#include "mlk/support/json.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace mlk::json {

namespace {

/// Nesting beyond this depth is rejected: untrusted input must not be able to
/// drive unbounded recursion (Rule 124).
constexpr int kMaxDepth = 128;

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Result<Value> run() {
        skipWs();
        Value v;
        MLK_TRYV(parseValue(v, 0));
        skipWs();
        if (pos_ != text_.size()) {
            return err(ErrorCode::ParseError,
                       "trailing characters after JSON value at offset " +
                           std::to_string(pos_));
        }
        return v;
    }

private:
    [[nodiscard]] bool eof() const { return pos_ >= text_.size(); }
    [[nodiscard]] char peek() const { return text_[pos_]; }

    void skipWs() {
        while (!eof()) {
            const char c = peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool consume(char c) {
        if (!eof() && peek() == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    Result<Ok> parseValue(Value& out, int depth) {
        if (depth > kMaxDepth) {
            return err(ErrorCode::ParseError, "JSON nesting depth exceeded");
        }
        skipWs();
        if (eof()) return err(ErrorCode::ParseError, "unexpected end of input");
        switch (peek()) {
            case '{': return parseObject(out, depth);
            case '[': return parseArray(out, depth);
            case '"': {
                std::string s;
                MLK_TRYV(parseString(s));
                out = Value(std::move(s));
                return ok();
            }
            case 't':
                if (match("true")) {
                    out = Value(true);
                    return ok();
                }
                return err(ErrorCode::ParseError, "invalid literal");
            case 'f':
                if (match("false")) {
                    out = Value(false);
                    return ok();
                }
                return err(ErrorCode::ParseError, "invalid literal");
            case 'n':
                if (match("null")) {
                    out = Value(nullptr);
                    return ok();
                }
                return err(ErrorCode::ParseError, "invalid literal");
            default: return parseNumber(out);
        }
    }

    [[nodiscard]] bool match(std::string_view kw) {
        if (text_.substr(pos_, kw.size()) == kw) {
            pos_ += kw.size();
            return true;
        }
        return false;
    }

    Result<Ok> parseNumber(Value& out) {
        const std::size_t start = pos_;
        if (!eof() && (peek() == '-' || peek() == '+')) ++pos_;
        bool isDouble = false;
        while (!eof()) {
            const char c = peek();
            if (c >= '0' && c <= '9') {
                ++pos_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                isDouble = isDouble || c == '.' || c == 'e' || c == 'E';
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) {
            return err(ErrorCode::ParseError,
                       "expected number at offset " + std::to_string(pos_));
        }
        const std::string numText{text_.substr(start, pos_ - start)};
        if (isDouble) {
            char* endp = nullptr;
            const double v = std::strtod(numText.c_str(), &endp);
            if (endp == nullptr || *endp != '\0') {
                return err(ErrorCode::ParseError, "malformed double: " + numText);
            }
            out = Value(v);
        } else {
            errno = 0;
            char* endp = nullptr;
            const long long v = std::strtoll(numText.c_str(), &endp, 10);
            if (endp == nullptr || *endp != '\0' || errno == ERANGE) {
                // Fall back to double for out-of-range integers.
                out = Value(std::strtod(numText.c_str(), nullptr));
            } else {
                out = Value(static_cast<int64_t>(v));
            }
        }
        return ok();
    }

    Result<Ok> parseString(std::string& out) {
        if (!consume('"')) {
            return err(ErrorCode::ParseError, "expected string");
        }
        out.clear();
        while (!eof()) {
            const char c = text_[pos_++];
            if (c == '"') return ok();
            if (c == '\\') {
                if (eof()) break;
                const char esc = text_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) {
                            return err(ErrorCode::ParseError,
                                       "truncated unicode escape");
                        }
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            const char h = text_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') {
                                code |= static_cast<unsigned>(h - '0');
                            } else if (h >= 'a' && h <= 'f') {
                                code |= static_cast<unsigned>(h - 'a' + 10);
                            } else if (h >= 'A' && h <= 'F') {
                                code |= static_cast<unsigned>(h - 'A' + 10);
                            } else {
                                return err(ErrorCode::ParseError,
                                           "bad unicode escape");
                            }
                        }
                        // Encode as UTF-8 (BMP only; surrogate pairs pass
                        // through as replacement chars — deterministic).
                        if (code < 0x80) {
                            out.push_back(static_cast<char>(code));
                        } else if (code < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out.push_back(
                                static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out.push_back(static_cast<char>(
                                0x80 | ((code >> 6) & 0x3F)));
                            out.push_back(
                                static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        return err(ErrorCode::ParseError,
                                   "unknown escape sequence");
                }
            } else {
                out.push_back(c);
            }
        }
        return err(ErrorCode::ParseError, "unterminated string");
    }

    Result<Ok> parseArray(Value& out, int depth) {
        consume('[');
        Array arr;
        skipWs();
        if (consume(']')) {
            out = Value(std::move(arr));
            return ok();
        }
        for (;;) {
            Value item;
            MLK_TRYV(parseValue(item, depth + 1));
            arr.push_back(std::move(item));
            skipWs();
            if (consume(',')) {
                skipWs();
                continue;
            }
            if (consume(']')) break;
            return err(ErrorCode::ParseError,
                       "expected ',' or ']' in array at offset " +
                           std::to_string(pos_));
        }
        out = Value(std::move(arr));
        return ok();
    }

    Result<Ok> parseObject(Value& out, int depth) {
        consume('{');
        Object obj;
        skipWs();
        if (consume('}')) {
            out = Value(std::move(obj));
            return ok();
        }
        for (;;) {
            skipWs();
            std::string key;
            MLK_TRYV(parseString(key));
            skipWs();
            if (!consume(':')) {
                return err(ErrorCode::ParseError,
                           "expected ':' after object key at offset " +
                               std::to_string(pos_));
            }
            Value v;
            MLK_TRYV(parseValue(v, depth + 1));
            obj.push_back(Member{
                std::move(key),
                std::shared_ptr<Value>(new Value(std::move(v)))});
            skipWs();
            if (consume(',')) continue;
            if (consume('}')) break;
            return err(ErrorCode::ParseError,
                       "expected ',' or '}' in object at offset " +
                           std::to_string(pos_));
        }
        out = Value(std::move(obj));
        return ok();
    }

    std::string_view text_;
    std::size_t pos_{0};
};

void escapeInto(const std::string& s, std::string& out) {
    out.push_back('"');
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void serializeDouble(double d, std::string& out) {
    if (d != d) {  // JSON has no NaN; canonical deterministic substitute
        out += "null";
        return;
    }
    if (std::isinf(d)) {
        out += d > 0 ? "1e308" : "-1e308";  // deterministic clamp
        return;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", d);
    out += buf;
}

void serializeImpl(const Value& v, std::string& out, int indent, int depth) {
    const bool pretty = indent >= 0;
    auto newline = [&](int level) {
        if (pretty) {
            out.push_back('\n');
            for (int i = 0; i < level; ++i) out += "  ";
        }
    };
    if (depth > kMaxDepth) return;  // defensive: cannot occur on parsed trees
    switch (v.type()) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += v.asBool() ? "true" : "false"; break;
        case Type::Int: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%lld",
                          static_cast<long long>(v.asInt()));
            out += buf;
            break;
        }
        case Type::Double: serializeDouble(v.asDouble(), out); break;
        case Type::String: escapeInto(v.asString(), out); break;
        case Type::Array: {
            const auto& arr = v.asArray();
            if (arr.empty()) {
                out += "[]";
                break;
            }
            out.push_back('[');
            for (std::size_t i = 0; i < arr.size(); ++i) {
                if (i != 0) out.push_back(',');
                newline(depth + 1);
                serializeImpl(arr[i], out, indent, depth + 1);
            }
            newline(depth);
            out.push_back(']');
            break;
        }
        case Type::Object: {
            const auto& obj = v.asObject();
            if (obj.empty()) {
                out += "{}";
                break;
            }
            out.push_back('{');
            for (std::size_t i = 0; i < obj.size(); ++i) {
                if (i != 0) out.push_back(',');
                newline(depth + 1);
                escapeInto(obj[i].key, out);
                out.push_back(':');
                if (pretty) out.push_back(' ');
                serializeImpl(*obj[i].value, out, indent, depth + 1);
            }
            newline(depth);
            out.push_back('}');
            break;
        }
    }
}

}  // namespace

Result<Value> parse(std::string_view text) { return Parser(text).run(); }

std::string serialize(const Value& v) {
    std::string out;
    serializeImpl(v, out, -1, 0);
    return out;
}

std::string serializePretty(const Value& v) {
    std::string out;
    serializeImpl(v, out, 2, 0);
    out.push_back('\n');
    return out;
}

}  // namespace mlk::json
