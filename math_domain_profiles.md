# MLK+ Math Domain Profiles

A Math Domain Profile is a versioned description of a domain's semantics,
capabilities, laws, numeric model, and approximation policy (Part 0). The
core compiler never hardcodes domain semantics (Rule 28).

## Schema

See `schemas/math_domain_profile.schema.json`. Profile JSON files live in
`profiles/<name>/profile.json`; regenerate with
`mlk-generate-profiles <dir>`.

## Capability flags

Part 0 defines 47 capabilities (HasSymbolicValues ... HasResumableEvaluation).
If a capability is absent the compiler must not assume the feature; if
present it must preserve observable semantics. Passes gate themselves:
e.g. `calculus.derivative_symbolic` refuses without HasDerivatives.

## Shipped profiles

| Profile | Focus | Notes |
|---|---|---|
| scalar_f64 | scalar IEEE double math | reassociation forbidden by default |
| tensor_f32_cpu | dense CPU tensors | fusion + autotuning capable |
| tensor_f16_gpu | GPU f16 tensors | backend capability only in MVP |
| symbolic_real | exact symbolic math | reassociation laws exact |
| calculus_real | symbolic + calculus | derivatives/integrals/limits |

FP reassociation is forbidden in every shipped profile by default
(Rules 33, 90): enabling it is a profile decision, reviewed and versioned.
