// mlk-poly bench verb (declarative in mlk_poly.cpp; Rule 27).
#pragma once

namespace polybench {

/// Runs the polyhedral runtime benchmark:
///   mlk-poly bench [--reps=R] [--warmup=W]
///                  [--suites=gemm,softmax,reducesum]
///                  [--json-out=F] [--workdir=D]
/// Every suite case runs five execution paths over the SAME seeded
/// inputs — Tier1 call baseline, Tier2 polyhedral walker, the C++ /
/// assembly native artifacts, and the CUDA artifact (nvcc; honest
/// skip rows when the toolchain probe fails, ULP-measured notes for
/// transcendental-policy modules) — with an exactness gate BEFORE any
/// timing (bit-exact for the device-exact op set; measured max-ULP
/// recorded for device-libm modules), then Rule 49 statistics
/// (median/min/max of reps timed runs after warmups).
int runBench(int argc, char** argv);

}  // namespace polybench
