# MLK+ Property System

## Lattice

Three-point tri-state lattice per property (Rule 22):

```
       Unknown        (no information — the default)
       /      \
     True    False     (definite facts)
       \      /
      conflict       (conservatively False + verifier diagnostic)
```

Unknown never silently becomes True. Only definite facts license rewrites.

## Properties

Commutative, Associative, Distributive, Idempotent, HasIdentity,
MonotonicIncreasing, MonotonicDecreasing, Periodic (+period payload),
Differentiable, Invertible, Positive, NonNegative, Bounded (+Interval
payload), Sparse, Symmetric, Contiguous, Pure, IntegerValued.

## Inference

`property.infer` derives facts from op class + operand domains + the Math
Domain Profile law declarations:

- Add commutes in fields; Associative is Unknown for FP unless the profile
  explicitly allows reassociation (Rule 33 — floating-point addition is not
  associative without a contract).
- MatMul is definitively NON-commutative (Rule 33).
- exp > 0 and monotone; sin/cos in [-1, 1] and periodic; tanh in [-1, 1] and
  monotone (spec Pass-3 fact examples).

## Consumers

Rewrite legality predicates, approximation gating, verifier consistency
checks, and the e-graph rule set all consume facts. Facts ride on values and
e-classes so equivalent forms inherit them.
