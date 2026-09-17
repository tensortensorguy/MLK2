// mlk-poly bench verb (declarative in mlk_poly.cpp; Rule 27).
#pragma once

namespace polybench {

/// Runs the polyhedral runtime benchmark:
///   mlk-poly bench [--reps=R] [--warmup=W]
///                  [--suites=gemm,softmax,reducesum]
///                  [--json-out=F] [--workdir=D]
/// Every suite case runs four execution paths over the SAME seeded
/// inputs — Tier1 call baseline, Tier2 polyhedral walker, and the C++
/// / assembly native artifacts — with a bit-exact gate (baseline vs
/// path, Rule 43 element semantics) BEFORE any timing, then Rule 49
/// statistics (median/min/max of reps timed runs after warmups).
int runBench(int argc, char** argv);

}  // namespace polybench
