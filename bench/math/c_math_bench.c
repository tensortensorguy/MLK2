/*
 * c_math_bench.c — plain C baselines for the MLK+ math benchmark suite.
 *
 * Hand-written C that a competent engineer would produce for each workload:
 * straightforward loops, no intrinsics, no framework. Compiled with
 * gcc -O3 -march=native (the standard "C baseline" configuration).
 *
 * Protocol (Rule 49): warmup 10, reps 30, min/median/stddev, seeded
 * golden-ratio low-discrepancy inputs identical to the MLK+ harness.
 * Threads: 2 (matching the executor's hardware_concurrency default) via
 * pthreads over disjoint ranges — the same decision MLK+'s executor makes.
 */
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N (1 << 20)
#define M_DIM 512
#define WARMUP 10
#define REPS 30
#define NTHREADS 2

static double x[N], y[N];
static double A[M_DIM * M_DIM], B[M_DIM * M_DIM], C[M_DIM * M_DIM];
static double lo_range = -3.0, hi_range = 3.0;

static void fill_low_discrepancy(double *v, long n, double lo, double hi) {
    const double golden = 0.61803398874989484820;
    for (long i = 0; i < n; ++i) {
        double f = fmod((double)i * golden, 1.0);
        v[i] = lo + (hi - lo) * f;
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef void (*kernel_fn)(long begin, long end);

static kernel_fn g_kernel;
static long g_size;

static void *thread_entry(void *arg) {
    long tid = (long)arg;
    long chunk = (g_size + NTHREADS - 1) / NTHREADS;
    long b = tid * chunk;
    long e = b + chunk < g_size ? b + chunk : g_size;
    if (b < g_size) g_kernel(b, e);
    return NULL;
}

static void run_parallel(kernel_fn k, long size) {
    g_kernel = k;
    g_size = size;
    if (size < 16384 || NTHREADS == 1) {
        k(0, size);
        return;
    }
    pthread_t th[NTHREADS];
    for (long t = 0; t < NTHREADS; ++t)
        pthread_create(&th[t], NULL, thread_entry, (void *)t);
    for (long t = 0; t < NTHREADS; ++t)
        pthread_join(th[t], NULL);
}

/* --- workloads: same math as the MLK+ graphs ---------------------------- */
static void k_sin_sq_3x(long b, long e) {
    for (long i = b; i < e; ++i) y[i] = sin(x[i] * x[i] + 3.0 * x[i]);
}
static void k_sin_sq_3x_fused(long b, long e) {
    for (long i = b; i < e; ++i)
        y[i] = (sin(x[i] * x[i] + 3.0 * x[i])) * 1.5 + 0.25;
}
static void k_tanh(long b, long e) {
    for (long i = b; i < e; ++i) y[i] = tanh(x[i]);
}
static void k_exp(long b, long e) {
    for (long i = b; i < e; ++i) y[i] = exp(x[i]);
}
static void k_derivative(long b, long e) {
    /* symbolic d/dx[x^2 sin(x)] = 2x sin(x) + x^2 cos(x), hand-derived */
    for (long i = b; i < e; ++i) {
        const double xi = x[i];
        y[i] = 2.0 * xi * sin(xi) + xi * xi * cos(xi);
    }
}
static void k_matmul_serial(long b, long e) {
    /* blocked ikj GEMM over rows [b, e) — same blocking family as MLK+ */
    const int64_t n = M_DIM, k = M_DIM;
    for (long i = b; i < e; ++i)
        for (int64_t p = 0; p < k; ++p) {
            const double aik = A[i * k + p];
            const double *bp = B + p * n;
            double *cp = C + i * n;
            for (int64_t j = 0; j < n; ++j) cp[j] += aik * bp[j];
        }
}

typedef struct {
    const char *name;
    kernel_fn fn;
    double lo, hi;
    int is_gemm;
} Wl;

int main(void) {
    const Wl wls[] = {
        {"sin_sq_3x", k_sin_sq_3x, -3.0, 3.0, 0},
        {"sin_sq_3x_fused", k_sin_sq_3x_fused, -3.0, 3.0, 0},
        {"tanh", k_tanh, -3.0, 3.0, 0},
        {"exp", k_exp, -2.0, 2.0, 0},
        {"derivative_x2_sin_x", k_derivative, -3.0, 3.0, 0},
        {"matmul_512", k_matmul_serial, -1.0, 1.0, 1},
    };
    const int nw = (int)(sizeof(wls) / sizeof(wls[0]));

    printf("{\"format\":\"c-math-bench\",\"protocol\":{\"warmup\":%d,"
           "\"reps\":%d,\"dtype\":\"f64\",\"threads\":%d},\"results\":[\n",
           WARMUP, REPS, NTHREADS);

    for (int w = 0; w < nw; ++w) {
        const long size = wls[w].is_gemm ? M_DIM : N;
        if (wls[w].is_gemm) {
            fill_low_discrepancy(A, (long)M_DIM * M_DIM, -1.0, 1.0);
            fill_low_discrepancy(B, (long)M_DIM * M_DIM, -1.0, 1.0);
            memset(C, 0, sizeof(C));
        } else {
            fill_low_discrepancy(x, N, wls[w].lo, wls[w].hi);
        }

        double samples[REPS];
        for (int r = 0; r < WARMUP; ++r) run_parallel(wls[w].fn, size);
        double sink = 0.0;
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now_ms();
            run_parallel(wls[w].fn, size);
            /* Consume the output: without this, dead-store elimination
             * removes the loops entirely (measured at 541 "GB/s"). A real
             * caller always reads the results. */
            const long step = N / 8;
            for (long i = 0; i < (wls[w].is_gemm ? M_DIM * M_DIM : N);
                 i += step)
                sink += wls[w].is_gemm ? C[i] : y[i];
            samples[r] = now_ms() - t0;
        }
        if (sink == 42.0) fprintf(stderr, "\n");  /* keeps sink live */
        /* min / median / stddev */
        double mn = samples[0], mean = 0.0;
        for (int r = 0; r < REPS; ++r) {
            if (samples[r] < mn) mn = samples[r];
            mean += samples[r];
        }
        mean /= REPS;
        for (int r = 0; r < REPS - 1; ++r)
            for (int s = r + 1; s < REPS; ++s)
                if (samples[s] < samples[r]) {
                    double t = samples[r];
                    samples[r] = samples[s];
                    samples[s] = t;
                }
        double med = samples[REPS / 2];
        double var = 0.0;
        for (int r = 0; r < REPS; ++r) var += (samples[r] - mean) * (samples[r] - mean);
        double sd = sqrt(var / REPS);

        double bytes = wls[w].is_gemm
                           ? (2.0 * M_DIM * M_DIM * M_DIM + M_DIM * M_DIM) * 8.0
                           : 2.0 * N * 8.0;
        printf("{\"name\":\"%s\",\"variant\":\"c_o3_native\",\"n\":%ld,"
               "\"dtype\":\"f64\",\"min_ms\":%.6f,\"median_ms\":%.6f,"
               "\"stddev_ms\":%.6f,\"gbps\":%.3f}%s\n",
               wls[w].name, (long)(wls[w].is_gemm ? M_DIM * M_DIM : N), mn,
               med, sd, bytes / 1e9 / (med / 1e3),
               w + 1 < nw ? "," : "");
    }
    printf("\n]}\n");
    return 0;
}
