# MLK+ Runtime Contracts

## ExecutionEngine

- `execute(graph, profile, contract, tier, inputs)` runs Tier 0 directly or
  an installed kernel, falling back to Tier 0 on any kernel failure
  (Rules 102, 115, 139).
- `requestAsyncCompile` compiles on a background thread; the evaluation
  thread keeps executing the current tier (Rule 11); installation is an
  atomic release-store; consumers acquire (Rule 117).
- Compilation always runs on a frozen snapshot (Rule 13).

## FallbackEngine

- `capture(state) -> GraphState` and `reconstruct(GraphState) -> state`
  implement Rule 102 (exact lower-tier state) and are roundtrip-tested.
- Per-site fallback counts throttle repeated failures (Rule 103); every
  fallback is a telemetry event (Rule 30).

## Telemetry

Structured, bounded ring; events: compile/tune attempts+failures, fallbacks,
guard failures, invalidations, budget violations, cache pressure, blacklist,
tier transitions, counters (Rule 157). No source graphs, user data, or
secrets.

## Realization cache

Complete versioned keys (Rule 57); entries validated on load — version
mismatches and malformed entries are rejected with InvalidArtifact, never
silently ignored (Rules 37, 124, 127).

## Safepoints

Long-running loops poll a cancellation token at bounded intervals
(`kSafepointPollIntervalIterations`); latency is a single atomic load
(Rules 107, 130).
