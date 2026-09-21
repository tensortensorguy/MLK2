# ADR-0009: Pass catalog truth — no registered no-ops

Date: 2026-09-21 (round 23)
Status: Accepted
Context: external repo review (round 22 fixed the P0 tier: e-graph
miscompile, coupled-bound lexmin/lexmax, fake ULP verifier, fake tiering,
tautological autotuner, failing lint). The same review flagged a second
tier: passes that are REGISTERED and documented but do nothing — the
"mostly stubs" pattern the project explicitly rejects.

## Context

The pass registry advertised 47 passes. Auditing each one against the
pipelines and its own documentation found three classes:

1. **Computed-and-dropped**: `cost.roofline` computed the roofline bound
   into a local and discarded it; `memory.liveness` computed a live set
   into a local and discarded it; `tensor.contraction_path` ran a
   matrix-chain DP over the WRONG dimensions (cols-only, never touching
   the contraction dims) and discarded the result.
2. **Registered no-ops with no legal domain**: `math.expression_balance`
   and `math.associative_flatten` claimed "realized through the e-graph"
   — the e-graph has identity and pow_square rules ONLY, no reassociation
   (and correctly so: FP reassociation is Rule 33-forbidden without an
   accuracy contract, and no shipped profile opens one).
3. **Placeholders around a real mechanism**: `schedule.fuse` /
   `schedule.region_extract` documented a fusion decision stage that is
   in fact realized by `tensor.fusion_find` (marks) + `lower.to_kernel_ir`
   (realization); `alias.infer` documented may-not-alias facts that are
   trivially true by IR construction (SSA-like fresh allocations; views
   are explicit LayoutTransform) and had no consumer.

Separately, fixing the catalog exposed two NEW certified miscompiles in
`math.identity_elim` (found by audit, confirmed by probe before fixing):

- **Int-constant miscompile**: `isConst(v, &f)` returns true for INT
  constants while leaving `*f` untouched, so `isConst(v, &f) && f == 0.0`
  read a default-initialized 0.0 and `add(x, int 5)` folded to `x`.
- **Inverted -0 gate**: `x + 0 -> x` gated on `signbit(CONSTANT)`, but
  the hazard is the OPERAND's sign: under `preserveNegativeZero = true`
  (the default), `x + (+0) -> x` is wrong for x = -0 (IEEE gives +0) —
  and the gate simultaneously blocked the ALWAYS-sound `x + (-0) -> x`.

## Decision

**Implement and wire what has a legal, bit-exact domain:**

1. `math.algebraic_simplify` — bit-exact FP sign/division identities:
   `sub(x, +0) -> x`, `div(x, 1) -> x`, `div(x, -1) -> neg(x)`,
   `neg(neg(x)) -> x`. Wired into Tiers 1/2/3 after math.canonicalize.
2. `math.strength_reduce` — `div(x, 2^k) -> mul(x, 2^-k)` for F64
   compile-time power-of-two constants (theorem + differential test;
   reciprocal-overflow guard). Wired into Tiers 1/2/3.
3. `cost.roofline` — the bound is published as a PerfCounter telemetry
   event (Rule 23's sanctioned channel), ceiled so positive bounds never
   read 0.
4. The autotuner prunes for real: `rooflineCandidateLowerBoundNs` caps
   achievable throughput by the candidate's threads/vector-width knobs
   (a sound upper bound on resources -> a valid per-candidate lower
   bound), the seed is measured first, and candidates whose provable
   minimum exceeds the best measured time (within the noise margin) are
   skipped with telemetry. The old code assigned ONE graph-level bound to
   every candidate and kept everything within 2x of it — an identity
   comparison that pruned nothing.
5. `tensor.contraction_path` — correct matrix-chain DP over the leaf
   matrices of actual chains, with the order recorded on the IR
   (`contraction_split`/`contraction_flops`/`contraction_flops_naive`
   attrs) and a `contraction_chains` telemetry event. Realization of a
   non-left-deep order reassociates FP matmuls and stays blocked until a
   Rule 33 accuracy contract exists — recorded honestly in the pass doc.

**Remove what has no legal domain or no consumer** (with this ADR as the
record): `math.expression_balance`, `math.associative_flatten`,
`schedule.fuse`, `schedule.region_extract`, `alias.infer`,
`memory.liveness`. Registry: 47 advertised -> 41 real + 7 previously
unlisted polyhedral passes = 48 (the committed registry had gone stale).

**Legality discipline**: typed constant probes (`isFpConst`/`isIntConst`)
replace the raw `isConst` in every identity check, closing the
int-constant-as-0.0 miscompile class; `identity_elim`'s Add gate is
operand-signed as above.

**Fix `dtypeBytes`**: C64 = 8, C128 = 16 (complex is a pair of component
floats; the 4/8 values under-counted every complex byte figure 2x).

**Tools**: `mlkc compile --telemetry[=path]` dumps the structured event
stream (pass/reason resolved through the emitting table);
`TelemetrySink::toJson(const SymbolTable*)` is the table-resolving
overload; `mlk-profile` is a real aggregator (counts + counter sums by
kind, `--fallbacks` / `--guard-failures` listing) over that stream —
previously it parsed nothing and printed help.

**Tests**: regression tests exist for every change above (typed-probe
int identities, operand-signed -0 gate + end-to-end -0 semantics,
algebraic identities, strength-reduce bit-exact differential incl.
subnormals, bound monotonicity + prune-fires telemetry, chain DP with
per-node attrs, complex dtypeBytes, measured superopt speedups); the
portable `fkwork` workroot replaces the machine-specific test paths.

## Consequences

- Every registered pass now either transforms the graph, publishes an
  analysis on a declared channel, or records a decision on the IR.
  "Registered" is once again a claim of function.
- The e-graph remains the sole owner of reassociation-shaped future work;
  it gains associative rules only together with the accuracy-contract
  gate that makes them legal.
- `memory.liveness` returns only with a consumer (buffer sharing), as an
  analysis with a declared output.
- No behavior change to bit-exactness anywhere: every new rewrite is
  bit-exact by theorem and differential test; every removed pass changed
  nothing by construction.
