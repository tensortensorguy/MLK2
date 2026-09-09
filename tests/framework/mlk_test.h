// MLK+ hermetic test harness (Rule 72: self-contained, reproducible; no
// network, no exceptions, no external framework). Failures are recorded;
// the process exits nonzero when any check failed.
#pragma once

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>

namespace mlk::test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int& failures() {
    static int f = 0;
    return f;
}

inline int& currentChecks() {
    static int c = 0;
    return c;
}

inline bool registerTest(const char* name, void (*fn)()) {
    registry().push_back(TestCase{name, fn});
    return true;
}

inline void fail(const char* file, int line, const char* what) {
    ++failures();
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, what);
}

inline std::string& currentTestName() {
    static std::string n;
    return n;
}

inline int runAll(const char* suite) {
    int ran = 0;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::set_terminate([]() {
        std::fprintf(stderr, "[abort] during test: %s\n",
                     currentTestName().c_str());
        std::abort();
    });
    for (const auto& t : registry()) {
        currentTestName() = t.name;
        std::printf("[%s] %s\n", suite, t.name);
        const int before = failures();
        currentChecks() = 0;
#ifdef MLK_TEST_CATCH
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[exception] in %s: %s\n",
                         currentTestName().c_str(), e.what());
            ++failures();
        } catch (...) {
            std::fprintf(stderr, "[exception] in %s: unknown\n",
                         currentTestName().c_str());
            ++failures();
        }
#else
        t.fn();
#endif
        currentTestName().clear();
        ++ran;
        if (failures() == before && currentChecks() == 0) {
            std::fprintf(stderr, "  WARN %s: no checks executed\n", t.name);
        }
    }
    std::printf("[%s] %d tests, %d failures\n", suite, ran, failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace mlk::test

#define MLK_TEST(suite, name)                                                 \
    static void mlk_test_##suite##_##name();                                  \
    static const bool mlk_reg_##suite##_##name =                              \
        ::mlk::test::registerTest(#suite "." #name,                           \
                                  &mlk_test_##suite##_##name);                \
    static void mlk_test_##suite##_##name()

#define MLK_CHECK(cond)                                                       \
    do {                                                                      \
        ++::mlk::test::currentChecks();                                       \
        if (!(cond)) ::mlk::test::fail(__FILE__, __LINE__, #cond);            \
    } while (false)

#define MLK_CHECK_EQ(a, b)                                                    \
    do {                                                                      \
        ++::mlk::test::currentChecks();                                       \
        if (!((a) == (b))) {                                                  \
            ::mlk::test::fail(__FILE__, __LINE__, #a " == " #b);              \
        }                                                                     \
    } while (false)

#define MLK_CHECK_NEAR(a, b, tol)                                             \
    do {                                                                      \
        ++::mlk::test::currentChecks();                                       \
        const double _a = (a);                                            \
        const double _b = (b);                                            \
        const double _t = (tol);                          \
        if (!(_a - _b <= _t && _b - _a <= _t)) {                              \
            ::mlk::test::fail(__FILE__, __LINE__, #a " ~= " #b);              \
        }                                                                     \
    } while (false)

// Suites define their own main via MLK_TEST_MAIN(suite). The build system
// may pass -DMLK_TEST_MAIN=\"name\" for suite naming; detect and route.
#if defined(MLK_TEST_MAIN)
#undef MLK_TEST_MAIN
#define MLK_TEST_MAIN_IMPL
#define MLK_TEST_MAIN(suite)                                                  \
    int main() { return ::mlk::test::runAll(suite); }
#else
#define MLK_TEST_MAIN(suite)                                                  \
    int main() { return ::mlk::test::runAll(suite); }
#endif
