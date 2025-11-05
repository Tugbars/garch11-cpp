/* High-Performance GARCH(1,1) Implementation
 * Implementation file with scalar and AVX2 optimized paths
 */

#include "garch11.h"
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdatomic.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __SSE2__
#include <xmmintrin.h>
#endif

/* ============================================================================
 * Denormal Protection (FTZ/DAZ)
 * ============================================================================ */

#ifdef __SSE2__
/**
 * Enable Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ)
 * Prevents performance degradation from denormal arithmetic
 * Call once at initialization
 */
static inline void enable_ftz_daz(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

// Auto-enable on library load
__attribute__((constructor)) static void garch_init(void)
{
    enable_ftz_daz();
}
#endif

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * Sanitize lambda for EWMA (clamp to sensible range)
 */
static inline double sanitize_lambda(double lambda)
{
    return (lambda > 0.0 && lambda < 1.0) ? lambda : 0.97;
}

/**
 * Sanitize r2 value for NaN/Inf (hot path - branchless)
 */
static inline __attribute__((always_inline)) double clean_r2(double x)
{
    // Branchless: if finite and positive, return x; else 0.0
    return (isfinite(x) && x > 0.0) ? x : 0.0;
}

/**
 * Compute dynamic epsilon based on variance scale (branchless)
 */
static inline double dynamic_eps(double level, const garch_config_t *cfg)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double e = cfg->eps_base * scale;
    return (e < cfg->eps_min) ? cfg->eps_min : e;
}

/**
 * FAST PATH: Clamp sigma2 without tracking (branch-free hot path)
 */
static inline __attribute__((always_inline)) double safe_sigma2_fast(double sigma2_raw, double level,
                                                                     const garch_config_t *cfg)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    return (sigma2_raw < eps) ? eps : sigma2_raw;
}

/**
 * TRACKING PATH: Clamp sigma2 with statistics (for diagnostics)
 * Uses atomic increment if C11 atomics available (thread-safe)
 */
static inline __attribute__((always_inline)) double safe_sigma2_track(double sigma2_raw, double level,
                                                                      const garch_config_t *cfg,
                                                                      garch_clamp_stats_t *stats)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    const bool clamped = (sigma2_raw < eps);
    if (clamped)
    {
        // Thread-safe atomic increment (C11)
        atomic_fetch_add_explicit((_Atomic int *)&stats->sigma2_clamps, 1, memory_order_relaxed);
    }
    return clamped ? eps : sigma2_raw;
}

/**
 * Dispatcher for safe_sigma2 (compatibility wrapper)
 */
static inline double safe_sigma2(double sigma2_raw, double level,
                                 const garch_config_t *cfg,
                                 garch_clamp_stats_t *stats)
{
    if (stats && cfg->track_clamps)
    {
        return safe_sigma2_track(sigma2_raw, level, cfg, stats);
    }
    else
    {
        return safe_sigma2_fast(sigma2_raw, level, cfg);
    }
}

/**
 * Compute single NLL term: log(sigma2) + r2/sigma2
 * Uses FMA for efficiency
 */
static inline __attribute__((always_inline)) double nll_term(double r2, double sigma2)
{
    double inv_sigma2 = 1.0 / sigma2;
    return fma(r2, inv_sigma2, log(sigma2));
}

/**
 * Compute derivative of NLL term w.r.t. sigma2: (1 - r2/sigma2) / sigma2
 */
static inline __attribute__((always_inline)) double d_nll_term_d_sigma2(double r2, double sigma2)
{
    double inv = 1.0 / sigma2;
    return inv * (1.0 - r2 * inv);
}

/* ============================================================================
 * Backcast Implementations (Cold Path)
 * ============================================================================ */

/**
 * Backcast: Simple mean
 */
static double backcast_mean(const double *restrict r2, int n)
{
    if (n <= 0)
        return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += (r2[i] > 0.0) ? r2[i] : 0.0;
    }
    return sum / (double)n;
}

/**
 * Backcast: Triangular (early emphasis)
 */
static double backcast_triangular(const double *restrict r2, int n)
{
    int tau = (n < 75) ? n : 75;
    if (tau <= 0)
        return 0.0;
    if (tau == 1)
        return (r2[0] > 0.0) ? r2[0] : 0.0;

    double denom = 0.5 * tau * (tau + 1);
    double sum = 0.0;
    for (int i = 0; i < tau; i++)
    {
        double w = (double)(tau - i) / denom;
        sum += w * ((r2[i] > 0.0) ? r2[i] : 0.0);
    }
    return (sum > 0.0) ? sum : 0.0;
}

/**
 * Backcast: EWMA
 */
static double backcast_ewma(const double *restrict r2, int n, double lambda)
{
    lambda = sanitize_lambda(lambda);
    int tau = (n < 75) ? n : 75;
    if (tau <= 0)
        return 0.0;
    if (tau == 1)
        return (r2[0] > 0.0) ? r2[0] : 0.0;

    double wsum = 0.0, sum = 0.0, w = 1.0;
    for (int i = 0; i < tau; i++)
    {
        sum += w * ((r2[i] > 0.0) ? r2[i] : 0.0);
        wsum += w;
        w *= lambda;
    }

    // Stability guard
    if (!(wsum > 0.0) || !isfinite(wsum) || !isfinite(sum))
    {
        return backcast_mean(r2, tau);
    }

    double bc = sum / wsum;
    return (bc > 0.0) ? bc : 0.0;
}

double garch_compute_backcast(const double *restrict r2, int n,
                              garch_backcast_method_t method,
                              double lambda)
{
    switch (method)
    {
    case GARCH_BACKCAST_MEAN:
        return backcast_mean(r2, n);
    case GARCH_BACKCAST_EWMA:
        return backcast_ewma(r2, n, lambda);
    case GARCH_BACKCAST_TRIANGULAR:
    default:
        return backcast_triangular(r2, n);
    }
}

/* ============================================================================
 * Scalar Implementations (Reference / Fallback)
 * ============================================================================ */

/**
 * Scalar NLL with backcast - optimized with software pipelining
 */
double garch_nll_bc(const garch_params_t *restrict params,
                    const garch_data_t *restrict data,
                    const garch_config_t *restrict config,
                    garch_clamp_stats_t *restrict stats)
{
    // Use defaults if config is NULL
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;

    // Alignment hint for better codegen (r2 should be 32-byte aligned)
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    // Validate
    if (n <= 0 || !garch_valid_params(params))
    {
        return INFINITY;
    }

    // Prefetch distance (in elements, not bytes)
    const int prefetch_elems = 64;

    // Unswitch loop based on tracking and sanitization
    const bool track = cfg->track_clamps && stats;
    const bool sanitize = cfg->sanitize_r2;

    if (!track)
    {
        // FAST PATH: No tracking
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast(s, s, cfg);

        // Precompute inv and log for software pipelining
        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];
        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        for (int i = 1; i < n; i++)
        {
            // Prefetch future r2 values
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            // Sanitize if needed
            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            // Compute next sigma2 (long latency)
            double s_next = omega + fma(alpha, r2_prev, beta * s);
            const double level = (s_next > s) ? s_next : s;
            s_next = safe_sigma2_fast(s_next, level, cfg);

            // Finish current NLL term with precomputed inv/ln
            s = s_next;
            inv = 1.0 / s;
            ln = log(s);
            nll += fma(r2_curr, inv, ln);
        }

        return nll;
    }
    else
    {
        // TRACKING PATH: Statistics enabled
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track(s, s, cfg, stats);

        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];
        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_next = omega + fma(alpha, r2_prev, beta * s);
            const double level = (s_next > s) ? s_next : s;
            s_next = safe_sigma2_track(s_next, level, cfg, stats);

            s = s_next;
            inv = 1.0 / s;
            ln = log(s);
            nll += fma(r2_curr, inv, ln);
        }

        return nll;
    }
}

/**
 * Scalar gradient with backcast - optimized
 */
void garch_gradient_bc(const garch_params_t *restrict params,
                       const garch_data_t *restrict data,
                       garch_gradient_t *restrict grad,
                       const garch_config_t *restrict config,
                       garch_clamp_stats_t *restrict stats)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;
    const double *restrict r2 = data->r2;
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        grad->d_omega = grad->d_alpha = grad->d_beta = 0.0;
        return;
    }

    const int prefetch_dist = 64;
    const bool track = cfg->track_clamps && stats;

    if (!track)
    {
        // FAST PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast(s, s, cfg);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        double dterm = d_nll_term_d_sigma2(r2[0], s);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            s = omega + fma(alpha, r2[i - 1], beta * s_prev);
            const double level = (s > s_prev) ? s : s_prev;
            s = safe_sigma2_fast(s, level, cfg);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2[i - 1]);
            d_beta = fma(beta, d_beta, s_prev);

            dterm = d_nll_term_d_sigma2(r2[i], s);
            grad_omega = fma(dterm, d_omega, grad_omega);
            grad_alpha = fma(dterm, d_alpha, grad_alpha);
            grad_beta = fma(dterm, d_beta, grad_beta);

            s_prev = s;
        }

        grad->d_omega = grad_omega;
        grad->d_alpha = grad_alpha;
        grad->d_beta = grad_beta;
    }
    else
    {
        // TRACKING PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track(s, s, cfg, stats);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        double dterm = d_nll_term_d_sigma2(r2[0], s);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            s = omega + fma(alpha, r2[i - 1], beta * s_prev);
            const double level = (s > s_prev) ? s : s_prev;
            s = safe_sigma2_track(s, level, cfg, stats);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2[i - 1]);
            d_beta = fma(beta, d_beta, s_prev);

            dterm = d_nll_term_d_sigma2(r2[i], s);
            grad_omega = fma(dterm, d_omega, grad_omega);
            grad_alpha = fma(dterm, d_alpha, grad_alpha);
            grad_beta = fma(dterm, d_beta, grad_beta);

            s_prev = s;
        }

        grad->d_omega = grad_omega;
        grad->d_alpha = grad_alpha;
        grad->d_beta = grad_beta;
    }
}

/**
 * Scalar NLL + gradient combined (most efficient for small n)
 */
double garch_nll_gradient_bc(const garch_params_t *restrict params,
                             const garch_data_t *restrict data,
                             garch_gradient_t *restrict grad,
                             const garch_config_t *restrict config,
                             garch_clamp_stats_t *restrict stats)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;
    const double *restrict r2 = data->r2;
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        grad->d_omega = grad->d_alpha = grad->d_beta = 0.0;
        return INFINITY;
    }

    const int prefetch_dist = 64;
    const bool track = cfg->track_clamps && stats;

    if (!track)
    {
        // FAST PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast(s, s, cfg);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2[0], inv, ln);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        double dterm = inv * (1.0 - r2[0] * inv);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            s = omega + fma(alpha, r2[i - 1], beta * s_prev);
            const double level = (s > s_prev) ? s : s_prev;
            s = safe_sigma2_fast(s, level, cfg);

            // NLL term
            inv = 1.0 / s;
            ln = log(s);
            nll += fma(r2[i], inv, ln);

            // Gradient update
            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2[i - 1]);
            d_beta = fma(beta, d_beta, s_prev);

            dterm = inv * (1.0 - r2[i] * inv);
            grad_omega = fma(dterm, d_omega, grad_omega);
            grad_alpha = fma(dterm, d_alpha, grad_alpha);
            grad_beta = fma(dterm, d_beta, grad_beta);

            s_prev = s;
        }

        grad->d_omega = grad_omega;
        grad->d_alpha = grad_alpha;
        grad->d_beta = grad_beta;
        return nll;
    }
    else
    {
        // TRACKING PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track(s, s, cfg, stats);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2[0], inv, ln);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        double dterm = inv * (1.0 - r2[0] * inv);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            s = omega + fma(alpha, r2[i - 1], beta * s_prev);
            const double level = (s > s_prev) ? s : s_prev;
            s = safe_sigma2_track(s, level, cfg, stats);

            inv = 1.0 / s;
            ln = log(s);
            nll += fma(r2[i], inv, ln);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2[i - 1]);
            d_beta = fma(beta, d_beta, s_prev);

            dterm = inv * (1.0 - r2[i] * inv);
            grad_omega = fma(dterm, d_omega, grad_omega);
            grad_alpha = fma(dterm, d_alpha, grad_alpha);
            grad_beta = fma(dterm, d_beta, grad_beta);

            s_prev = s;
        }

        grad->d_omega = grad_omega;
        grad->d_alpha = grad_alpha;
        grad->d_beta = grad_beta;
        return nll;
    }
}

/* ============================================================================
 * AVX2 Optimized Implementations
 * ============================================================================ */

#ifdef __AVX2__

/**
 * AVX2 NLL + Gradient (vectorized Jacobian) - Optimized
 *
 * Key optimizations:
 * - Vectorize {d_omega, d_alpha, d_beta} in parallel using AVX2
 * - Reduce shuffle overhead with blend instructions
 * - Loop unswitching for tracking vs fast path
 * - 4x unrolling for better ILP
 */
double garch_nll_gradient_bc_avx2(const garch_params_t *restrict params,
                                  const garch_data_t *restrict data,
                                  garch_gradient_t *restrict grad,
                                  const garch_config_t *restrict config,
                                  garch_clamp_stats_t *restrict stats)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;
    const double *restrict r2 = data->r2;
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        grad->d_omega = grad->d_alpha = grad->d_beta = 0.0;
        return INFINITY;
    }

    const int prefetch_dist = 64;
    const bool track = cfg->track_clamps && stats;

    // AVX2 constants (reduce redundant loads)
    const __m256d ZERO = _mm256_setzero_pd();
    const __m256d ONE_VEC = _mm256_set1_pd(1.0);
    const __m256d beta_vec = _mm256_set1_pd(beta);

    if (!track)
    {
        // ===== FAST PATH: No tracking =====
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast(s, s, cfg);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2[0], inv, ln);

        // Vectorize partials: [d_omega, d_alpha, d_beta, 0]
        __m256d d_params = _mm256_set_pd(0.0, bc, bc, 1.0);

        double dterm = inv * (1.0 - r2[0] * inv);
        __m256d grad_vec = _mm256_mul_pd(_mm256_set1_pd(dterm), d_params);

        double s_prev = s;

        // Main loop: 4x unrolled
        int i = 1;
        for (; i < n - 3; i += 4)
        {
            // ===== Iteration 1 =====
            {
                if (i + prefetch_dist < n)
                {
                    __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
                }

                double s_next = omega + fma(alpha, r2[i - 1], beta * s_prev);
                const double level = (s_next > s_prev) ? s_next : s_prev;
                s_next = safe_sigma2_fast(s_next, level, cfg);

                // NLL
                inv = 1.0 / s_next;
                ln = log(s_next);
                nll += fma(r2[i], inv, ln);

                // Vectorized gradient update (reduce shuffle overhead)
                // Build updates = [1.0, r2[i-1], s_prev, 0] using blends
                __m256d updates = ZERO;
                updates = _mm256_blend_pd(updates, ONE_VEC, 0x1);                   // Lane 0 = 1.0
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(r2[i - 1]), 0x2); // Lane 1
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(s_prev), 0x4);    // Lane 2

                d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

                dterm = inv * (1.0 - r2[i] * inv);
                grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

                s_prev = s_next;
            }

            // ===== Iteration 2 =====
            {
                double s_next = omega + fma(alpha, r2[i], beta * s_prev);
                const double level = (s_next > s_prev) ? s_next : s_prev;
                s_next = safe_sigma2_fast(s_next, level, cfg);

                inv = 1.0 / s_next;
                ln = log(s_next);
                nll += fma(r2[i + 1], inv, ln);

                __m256d updates = ZERO;
                updates = _mm256_blend_pd(updates, ONE_VEC, 0x1);
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(r2[i]), 0x2);
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(s_prev), 0x4);

                d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

                dterm = inv * (1.0 - r2[i + 1] * inv);
                grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

                s_prev = s_next;
            }

            // ===== Iteration 3 =====
            {
                double s_next = omega + fma(alpha, r2[i + 1], beta * s_prev);
                const double level = (s_next > s_prev) ? s_next : s_prev;
                s_next = safe_sigma2_fast(s_next, level, cfg);

                inv = 1.0 / s_next;
                ln = log(s_next);
                nll += fma(r2[i + 2], inv, ln);

                __m256d updates = ZERO;
                updates = _mm256_blend_pd(updates, ONE_VEC, 0x1);
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(r2[i + 1]), 0x2);
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(s_prev), 0x4);

                d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

                dterm = inv * (1.0 - r2[i + 2] * inv);
                grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

                s_prev = s_next;
            }

            // ===== Iteration 4 =====
            {
                double s_next = omega + fma(alpha, r2[i + 2], beta * s_prev);
                const double level = (s_next > s_prev) ? s_next : s_prev;
                s_next = safe_sigma2_fast(s_next, level, cfg);

                inv = 1.0 / s_next;
                ln = log(s_next);
                nll += fma(r2[i + 3], inv, ln);

                __m256d updates = ZERO;
                updates = _mm256_blend_pd(updates, ONE_VEC, 0x1);
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(r2[i + 2]), 0x2);
                updates = _mm256_blend_pd(updates, _mm256_set1_pd(s_prev), 0x4);

                d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

                dterm = inv * (1.0 - r2[i + 3] * inv);
                grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

                s_prev = s_next;
            }
        }

        // Remainder loop
        for (; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            double s_next = omega + fma(alpha, r2[i - 1], beta * s_prev);
            const double level = (s_next > s_prev) ? s_next : s_prev;
            s_next = safe_sigma2_fast(s_next, level, cfg);

            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2[i], inv, ln);

            __m256d updates = ZERO;
            updates = _mm256_blend_pd(updates, ONE_VEC, 0x1);
            updates = _mm256_blend_pd(updates, _mm256_set1_pd(r2[i - 1]), 0x2);
            updates = _mm256_blend_pd(updates, _mm256_set1_pd(s_prev), 0x4);

            d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

            dterm = inv * (1.0 - r2[i] * inv);
            grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

            s_prev = s_next;
        }

        // Extract gradient components
        double grad_array[4] __attribute__((aligned(32)));
        _mm256_store_pd(grad_array, grad_vec);

        grad->d_omega = grad_array[0];
        grad->d_alpha = grad_array[1];
        grad->d_beta = grad_array[2];

        return nll;
    }
    else
    {
        // ===== TRACKING PATH =====
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track(s, s, cfg, stats);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2[0], inv, ln);

        __m256d d_params = _mm256_set_pd(0.0, bc, bc, 1.0);

        double dterm = inv * (1.0 - r2[0] * inv);
        __m256d grad_vec = _mm256_mul_pd(_mm256_set1_pd(dterm), d_params);

        double s_prev = s;

        // Simplified loop (tracking is cold path, less aggressive optimization)
        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            double s_next = omega + fma(alpha, r2[i - 1], beta * s_prev);
            const double level = (s_next > s_prev) ? s_next : s_prev;
            s_next = safe_sigma2_track(s_next, level, cfg, stats);

            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2[i], inv, ln);

            __m256d updates = ZERO;
            updates = _mm256_blend_pd(updates, ONE_VEC, 0x1);
            updates = _mm256_blend_pd(updates, _mm256_set1_pd(r2[i - 1]), 0x2);
            updates = _mm256_blend_pd(updates, _mm256_set1_pd(s_prev), 0x4);

            d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

            dterm = inv * (1.0 - r2[i] * inv);
            grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

            s_prev = s_next;
        }

        double grad_array[4] __attribute__((aligned(32)));
        _mm256_store_pd(grad_array, grad_vec);

        grad->d_omega = grad_array[0];
        grad->d_alpha = grad_array[1];
        grad->d_beta = grad_array[2];

        return nll;
    }
}

/**
 * AVX2 NLL only (no gradient) - Optimized
 */
double garch_nll_bc_avx2(const garch_params_t *restrict params,
                         const garch_data_t *restrict data,
                         const garch_config_t *restrict config,
                         garch_clamp_stats_t *restrict stats)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;
    const double *restrict r2 = data->r2;
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        return INFINITY;
    }

    const int prefetch_dist = 64;
    const bool track = cfg->track_clamps && stats;

    if (!track)
    {
        // FAST PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast(s, s, cfg);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2[0], inv, ln);

        // 4x unrolled
        int i = 1;
        for (; i < n - 3; i += 4)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            double s1 = omega + fma(alpha, r2[i - 1], beta * s);
            const double l1 = (s1 > s) ? s1 : s;
            s1 = safe_sigma2_fast(s1, l1, cfg);
            inv = 1.0 / s1;
            ln = log(s1);
            nll += fma(r2[i], inv, ln);

            double s2 = omega + fma(alpha, r2[i], beta * s1);
            const double l2 = (s2 > s1) ? s2 : s1;
            s2 = safe_sigma2_fast(s2, l2, cfg);
            inv = 1.0 / s2;
            ln = log(s2);
            nll += fma(r2[i + 1], inv, ln);

            double s3 = omega + fma(alpha, r2[i + 1], beta * s2);
            const double l3 = (s3 > s2) ? s3 : s2;
            s3 = safe_sigma2_fast(s3, l3, cfg);
            inv = 1.0 / s3;
            ln = log(s3);
            nll += fma(r2[i + 2], inv, ln);

            double s4 = omega + fma(alpha, r2[i + 2], beta * s3);
            const double l4 = (s4 > s3) ? s4 : s3;
            s4 = safe_sigma2_fast(s4, l4, cfg);
            inv = 1.0 / s4;
            ln = log(s4);
            nll += fma(r2[i + 3], inv, ln);

            s = s4;
        }

        // Remainder
        for (; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            double s_next = omega + fma(alpha, r2[i - 1], beta * s);
            const double level = (s_next > s) ? s_next : s;
            s_next = safe_sigma2_fast(s_next, level, cfg);
            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2[i], inv, ln);
            s = s_next;
        }

        return nll;
    }
    else
    {
        // TRACKING PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track(s, s, cfg, stats);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2[0], inv, ln);

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_dist < n)
            {
                __builtin_prefetch(&r2[i + prefetch_dist], 0, 3);
            }

            double s_next = omega + fma(alpha, r2[i - 1], beta * s);
            const double level = (s_next > s) ? s_next : s;
            s_next = safe_sigma2_track(s_next, level, cfg, stats);
            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2[i], inv, ln);
            s = s_next;
        }

        return nll;
    }
}

#endif /* __AVX2__ */

/* ============================================================================
 * Cold Path: Transform and Utilities
 * ============================================================================ */

void garch_transform_bc(const garch_params_t *restrict params,
                        const garch_data_t *restrict data,
                        double *restrict out_sigma2,
                        const garch_config_t *restrict config,
                        garch_clamp_stats_t *restrict stats)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    double *restrict sigma2 = (double *)__builtin_assume_aligned(out_sigma2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        return;
    }

    const bool sanitize = cfg->sanitize_r2;

    double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];
    double s0 = omega + (alpha + beta) * bc;
    sigma2[0] = safe_sigma2(s0, s0, cfg, stats);

    for (int i = 1; i < n; i++)
    {
        double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
        double s = omega + fma(alpha, r2_prev, beta * sigma2[i - 1]);
        const double level = (s > sigma2[i - 1]) ? s : sigma2[i - 1];
        sigma2[i] = safe_sigma2(s, level, cfg, stats);
    }
}

double garch_get_last_sigma2(const garch_params_t *restrict params,
                             const garch_data_t *restrict data,
                             const garch_config_t *restrict config)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        return 0.0;
    }

    const bool sanitize = cfg->sanitize_r2;

    double sigma2 = omega + (alpha + beta) * bc;
    sigma2 = safe_sigma2_fast(sigma2, sigma2, cfg);

    for (int i = 1; i < n; i++)
    {
        double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
        double s_next = omega + fma(alpha, r2_prev, beta * sigma2);
        const double level = (s_next > sigma2) ? s_next : sigma2;
        sigma2 = safe_sigma2_fast(s_next, level, cfg);
    }

    return sigma2;
}

double garch_nll_from_path(const double *restrict r2,
                           const double *restrict sigma2,
                           int n,
                           const garch_config_t *restrict config,
                           garch_clamp_stats_t *restrict stats)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    double nll = 0.0;
    for (int i = 0; i < n; i++)
    {
        double s = safe_sigma2(sigma2[i], sigma2[i], cfg, stats);
        nll += nll_term(r2[i], s);
    }
    return nll;
}

/* ============================================================================
 * Prediction / Forecasting
 * ============================================================================ */

void garch_predict(const garch_params_t *restrict params,
                   double last_sigma2,
                   double last_r2,
                   const double *restrict randn_shocks,
                   int n,
                   double *restrict out_sigma2,
                   double *restrict out_returns,
                   const garch_config_t *restrict config)
{
    garch_config_t default_cfg = garch_default_config();
    const garch_config_t *cfg = config ? config : &default_cfg;

    const double omega = params->omega;
    const double alpha = params->alpha;
    const double beta = params->beta;

    // Alignment hints
    const double *restrict shocks = (const double *)__builtin_assume_aligned(randn_shocks, 32);
    double *restrict sigma2 = (double *)__builtin_assume_aligned(out_sigma2, 32);
    double *restrict returns = (double *)__builtin_assume_aligned(out_returns, 32);

    if (n <= 0 || !garch_valid_params(params))
    {
        return;
    }

    // Sanitize inputs
    if (!isfinite(last_sigma2) || last_sigma2 < 0.0)
        last_sigma2 = 0.0;
    if (!isfinite(last_r2))
        last_r2 = 0.0;

    // First step
    double s0 = omega + fma(alpha, last_r2, beta * last_sigma2);
    const double l0 = (omega > s0) ? omega : s0;
    s0 = safe_sigma2_fast(s0, l0, cfg);
    sigma2[0] = s0;

    double z0 = isfinite(shocks[0]) ? shocks[0] : 0.0;
    returns[0] = z0 * sqrt(s0);

    // Subsequent steps
    for (int i = 1; i < n; i++)
    {
        double zi = isfinite(shocks[i]) ? shocks[i] : 0.0;
        double r_prev_sq = returns[i - 1] * returns[i - 1];
        double s = omega + fma(alpha, r_prev_sq, beta * sigma2[i - 1]);
        const double level = (s > sigma2[i - 1]) ? s : sigma2[i - 1];
        s = safe_sigma2_fast(s, level, cfg);
        sigma2[i] = s;
        returns[i] = zi * sqrt(s);
    }
}