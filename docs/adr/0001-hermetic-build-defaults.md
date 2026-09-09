# ADR-0001: Hermetic builds by default

Status: Accepted

## Context
Rule 160 requires hermetic builds and pinned dependencies. Network fetches
(FetchContent) break offline reproducibility and CI determinism.

## Decision
All third-party dependencies default OFF (cmake/dependencies.cmake). The
default build uses the built-in minimal JSON DOM (mlk/support/json.h) and the
built-in test harness (tests/framework/mlk_test.h). GoogleTest,
nlohmann_json, LLVM, and Z3 remain optional flags with pinned URLs when
enabled.

## Consequences
No network needed to build/test; optional integrations exercised in optional
CI jobs.
