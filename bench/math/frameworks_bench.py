#!/usr/bin/env python3
"""frameworks_bench.py — torch.compile / JAX baselines for the MLK+ math
benchmark suite (bench/math/).

Identical workloads, seeds, ranges, protocol (Rule 49: warmup, reps,
min/median/stddev) as the MLK+ harness (math_bench.cpp) and the C baseline
(c_math_bench.c). f64 everywhere; JAX x64 mode enabled (MLK+ and C run
f64 — comparing f32 would be dishonest).

Outputs the same JSON schema so make_report.py can merge the three.
"""
import json
import statistics
import sys
import time

import numpy as np

N = 1 << 20
M = 512
WARMUP = 10
REPS = 30
GOLDEN = 0.61803398874989484820


def fill_low_discrepancy(n, lo, hi):
    idx = np.arange(n, dtype=np.float64)
    frac = np.mod(idx * GOLDEN, 1.0)
    return lo + (hi - lo) * frac


def now_ms():
    return time.perf_counter() * 1e3


def stats(samples):
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def bench(fn, warmup=WARMUP, reps=REPS, consume=None):
    for _ in range(warmup):
        out = fn()
        if consume is not None:
            consume(out)
    samples = []
    for _ in range(reps):
        t0 = now_ms()
        out = fn()
        if consume is not None:
            consume(out)
        samples.append(now_ms() - t0)
    return stats(samples)


def rec(name, variant, n, s, gbps, compile_ms=None):
    r = {"name": name, "variant": variant, "n": n, "dtype": "f64",
         "min_ms": s["min_ms"], "median_ms": s["median_ms"],
         "stddev_ms": s["stddev_ms"], "gbps": gbps}
    if compile_ms is not None:
        r["compile_ms"] = compile_ms
    return r


def run_torch():
    import torch
    torch.set_num_threads(2)
    results = []
    b_elem = 2.0 * N * 8.0

    def consume(t):
        return torch.sum(t).item()

    def gbps(n_bytes, med):
        return n_bytes / 1e9 / (med / 1e3)

    def timed(name, fn, n_bytes, compiled, compile_ms=None, is_gemm=False):
        s = bench(fn, consume=consume)
        return rec(name, "torch_compile" if compiled else "torch_eager",
                   M * M if is_gemm else N, s, gbps(n_bytes, s["median_ms"]),
                   compile_ms=compile_ms)

    def compiled_variant(name, make, x, n_bytes, is_gemm=False):
        t0 = now_ms()
        cf = torch.compile(make)
        _ = consume(cf(x))
        ct = now_ms() - t0
        return timed(name, lambda: cf(x), n_bytes, True, ct, is_gemm)

    x = torch.from_numpy(fill_low_discrepancy(N, -3.0, 3.0))
    w1 = lambda v: torch.sin(v * v + 3.0 * v)
    results.append(timed("sin_sq_3x", lambda: w1(x), b_elem, False))
    results.append(compiled_variant("sin_sq_3x", w1, x, b_elem))

    w2 = lambda v: torch.sin(v * v + 3.0 * v) * 1.5 + 0.25
    results.append(timed("sin_sq_3x_fused", lambda: w2(x), b_elem, False))
    results.append(compiled_variant("sin_sq_3x_fused", w2, x, b_elem))

    results.append(timed("tanh", lambda: torch.tanh(x), b_elem, False))
    results.append(compiled_variant("tanh", torch.tanh, x, b_elem))

    x2 = torch.from_numpy(fill_low_discrepancy(N, -2.0, 2.0))
    results.append(timed("exp", lambda: torch.exp(x2), b_elem, False))
    results.append(compiled_variant("exp", torch.exp, x2, b_elem))

    # derivative via autograd: f(x) = x^2 sin(x), d/dx over the tensor
    def dfin(v):
        v = v.detach().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad((v * v * torch.sin(v)).sum(), v)
        return g
    xd = torch.from_numpy(fill_low_discrepancy(N, -3.0, 3.0))
    results.append(timed("derivative_x2_sin_x", lambda: dfin(xd), b_elem,
                         False))
    results.append(compiled_variant("derivative_x2_sin_x", dfin, xd, b_elem))

    # matmul
    A = torch.from_numpy(fill_low_discrepancy(M * M, -1.0, 1.0)).reshape(M, M)
    Bm = torch.from_numpy(fill_low_discrepancy(M * M, -1.0, 1.0)).reshape(M, M)
    b_gemm = (2.0 * M * M * M + M * M) * 8.0
    results.append(timed("matmul_512", lambda: torch.matmul(A, Bm), b_gemm,
                         False, is_gemm=True))
    results.append(compiled_variant("matmul_512", lambda v: torch.matmul(A, Bm),
                                    A, b_gemm, is_gemm=True))

    return results


def run_jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    results = []
    b = 2.0 * N * 8.0

    x = jnp.asarray(fill_low_discrepancy(N, -3.0, 3.0))
    xf = lambda v: jnp.sin(v * v + 3.0 * v)
    s = bench(lambda: xf(x), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("sin_sq_3x", "jax_jit", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))

    jit_fn = jax.jit(xf)
    t0 = now_ms()
    _ = float(jax.block_until_ready(jnp.sum(jit_fn(x))))
    ct = now_ms() - t0
    s = bench(lambda: jit_fn(x), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("sin_sq_3x", "jax_jit_compiled", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3), compile_ms=ct))

    xf2 = lambda v: jnp.sin(v * v + 3.0 * v) * 1.5 + 0.25
    s = bench(lambda: xf2(x), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("sin_sq_3x_fused", "jax_jit", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))
    jit2 = jax.jit(xf2)
    s = bench(lambda: jit2(x), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("sin_sq_3x_fused", "jax_jit_compiled", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))

    s = bench(lambda: jnp.tanh(x),
              consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("tanh", "jax_jit", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))
    jit3 = jax.jit(jnp.tanh)
    s = bench(lambda: jit3(x), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("tanh", "jax_jit_compiled", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))

    x2 = jnp.asarray(fill_low_discrepancy(N, -2.0, 2.0))
    s = bench(lambda: jnp.exp(x2),
              consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("exp", "jax_jit", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))
    jit4 = jax.jit(jnp.exp)
    s = bench(lambda: jit4(x2), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("exp", "jax_jit_compiled", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))

    # derivative: per-element grad via vmap(grad)
    def f(v):
        return v * v * jnp.sin(v)
    grad_f = jax.vmap(jax.grad(f))
    xd = jnp.asarray(fill_low_discrepancy(N, -3.0, 3.0))
    s = bench(lambda: grad_f(xd),
              consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("derivative_x2_sin_x", "jax_jit", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3)))
    jit5 = jax.jit(grad_f)
    t0 = now_ms()
    _ = float(jax.block_until_ready(jnp.sum(jit5(xd))))
    ct = now_ms() - t0
    s = bench(lambda: jit5(xd), consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("derivative_x2_sin_x", "jax_jit_compiled", N, s,
                       b / 1e9 / (s["median_ms"] / 1e3), compile_ms=ct))

    A = jnp.asarray(fill_low_discrepancy(M * M, -1.0, 1.0).reshape(M, M))
    Bm = jnp.asarray(fill_low_discrepancy(M * M, -1.0, 1.0).reshape(M, M))
    bb = (2.0 * M * M * M + M * M) * 8.0
    s = bench(lambda: A @ Bm,
              consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("matmul_512", "jax_jit", M * M, s,
                       bb / 1e9 / (s["median_ms"] / 1e3)))
    jit6 = jax.jit(lambda a, b: a @ b)
    s = bench(lambda: jit6(A, Bm),
              consume=lambda t: float(jax.block_until_ready(jnp.sum(t))))
    results.append(rec("matmul_512", "jax_jit_compiled", M * M, s,
                       bb / 1e9 / (s["median_ms"] / 1e3)))
    return results


def main():
    out = {"format": "frameworks-math-bench",
           "protocol": {"warmup": WARMUP, "reps": REPS, "dtype": "f64"},
           "results": []}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "torch"):
        out["results"] += run_torch()
    if which in ("all", "jax"):
        out["results"] += run_jax()
    print(json.dumps(out, indent=None))


if __name__ == "__main__":
    main()
