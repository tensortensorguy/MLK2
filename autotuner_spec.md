# MLK+ Autotuner Specification

The autotuner is a **mathematical realization engine**: it searches for the
fastest physically executable realization that preserves mathematical
meaning, properties, and error bounds (realization spec §1).

## Search space (Rule 54)

Declarative and serializable: parameter names, domains, priors
(`mlk/autotune/search_space.h`). No ad-hoc mutation logic inside benchmark
loops. Elementwise graphs search vector_width/unroll/parallel; matmul graphs
add tile_m/tile_n/tile_k.

## Protocol (Rules 49, 56, 59)

- warmup + repeated measurement, min/median selection, stddev tracking;
- realistic seeded inputs (sin-composed patterns) — never all-zero buffers;
- noise filter: candidates whose relative stddev exceeds the noise floor are
  rejected; ties prefer simpler/lower-compile-cost realizations.

## Correctness precedes benchmarking (Rule 58)

Every candidate is verified against the Tier 0 oracle within the accuracy
contract BEFORE any timing. Unverified candidates are rejected without
benchmark promotion.

## Cost model (Rule 55)

Roofline lower bounds prune candidates before measurement:
time >= max(flops/peak, bytes/bandwidth).

## Caching (Rule 57)

Winning realizations are cached with COMPLETE keys: graph hash, property
facts hash, shape bucket, accuracy contract hash, profile version, compiler
version, pass-pipeline hash, superoptimizer version, hardware fingerprint.
Incomplete keys are correctness bugs. Entries are versioned and validated on
load (Rules 37, 124, 127).
