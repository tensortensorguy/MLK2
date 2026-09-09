# MLK+ Realization Cache

Caches **realizations, not just configs** (realization spec §17): strategy,
schedule, kernel hash, proof reference, and measured result.

## Key (Rule 57 — complete, or it's a bug)

graph hash; property facts hash; shape bucket; dtype/domain; layout
constraints; accuracy contract hash; profile version; compiler version;
pass-pipeline hash; superoptimizer version; hardware fingerprint; runtime
config hash.

## Values

Versioned JSON entries (`realization.schema.json`): format version, key,
strategy, kernel hash, schedule map, proof reference, measured ms.

## Trust boundary

Entries are untrusted (Rule 124): load validates format version and key
completeness; mismatches are rejected with InvalidArtifact (Rules 37, 127).
The in-memory cache is an open-addressing map keyed by the total hash
(Rule 17).
