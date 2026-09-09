# MLK+ Portable Math Bytecode

The portable executable form of a lowered region is the KernelModule JSON
format (`format: "mlk-kernel"`, version `kBytecodeFormatVersion`):

- `buffers`: named buffers with dtype, input/output flags, element counts,
  alignment;
- `nodes`: flat node array (loop/compute/load/store/alloc/copy/barrier/
  trace/guard/call) with explicit children links;
- `schedule`: declarative parameters (tile_m/n/k, vector_width, unroll,
  parallel, fallback_tier0 marker — Rule 100).

Node hashes + format version make artifacts cacheable and invalidatable
(Rules 24, 37). Artifacts are untrusted: the loader validates format
version and structure, and rejects mismatches with telemetry (Rules 124,
127).
