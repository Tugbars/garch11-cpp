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
 *
 * MXCSR is per-thread, so this must be called from each thread.
 */
void garch_enable_ftz_daz(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

// Auto-enable on library load (main thread only)
__attribute__((constructor)) static void garch_init(void)
{
    garch_enable_ftz_daz();
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
 * Returns clamped value
 */
static inline __attribute__((always_inline)) double safe_sigma2_fast(double sigma2_raw, double level,
                                                                     const garch_config_t *cfg)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    return (sigma2_raw < eps) ? eps : sigma2_raw;
}

/**
 * FAST PATH with clamp detection: Returns both clamped value and whether clamping occurred
 */
static inline __attribute__((always_inline)) double safe_sigma2_fast_detect(double sigma2_raw, double level,
                                                                            const garch_config_t *cfg,
                                                                            bool *clamped)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    *clamped = (sigma2_raw < eps);
    return (*clamped) ? eps : sigma2_raw;
}

/**
 * TRACKING PATH: Clamp sigma2 with statistics (for diagnostics)
 * Uses atomic increment for thread safety (sigma2_clamps is atomic_int)
 * Returns both clamped value and whether clamping occurred
 */
static inline __attribute__((always_inline)) double safe_sigma2_track_detect(double sigma2_raw, double level,
                                                                             const garch_config_t *cfg,
                                                                             garch_clamp_stats_t *stats,
                                                                             bool *clamped)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    *clamped = (sigma2_raw < eps);
    if (*clamped)
    {
        // Thread-safe atomic increment (no cast needed - field is atomic_int)
        atomic_fetch_add_explicit(&stats->sigma2_clamps, 1, memory_order_relaxed);
    }
    return (*clamped) ? eps : sigma2_raw;
}

/**
 * TRACKING PATH: Clamp sigma2 with statistics (for diagnostics)
 * Uses atomic increment for thread safety (sigma2_clamps is atomic_int)
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
        // Thread-safe atomic increment (no cast needed - field is atomic_int)
        atomic_fetch_add_explicit(&stats->sigma2_clamps, 1, memory_order_relaxed);
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
 * 1B) BATCH CLAMP STATS - Local counter helper
 * ============================================================================ */

/**
 * TRACKING PATH with local counter (no atomics in hot loop)
 * Thread-local clamp counting for batched flush
 */
static inline __attribute__((always_inline)) double safe_sigma2_track_local(
    double sigma2_raw, double level,
    const garch_config_t *cfg,
    int *local_clamps)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    const bool clamped = (sigma2_raw < eps);
    *local_clamps += clamped; // Simple increment, no atomic
    return clamped ? eps : sigma2_raw;
}

/**
 * TRACKING PATH with local counter + detection
 */
static inline __attribute__((always_inline)) double safe_sigma2_track_local_detect(
    double sigma2_raw, double level,
    const garch_config_t *cfg,
    int *local_clamps,
    bool *clamped)
{
    const double scale = (level > 1.0) ? level : 1.0;
    const double eps = (cfg->eps_base * scale < cfg->eps_min) ? cfg->eps_min : cfg->eps_base * scale;
    *clamped = (sigma2_raw < eps);
    *local_clamps += *clamped; // Simple increment
    return (*clamped) ? eps : sigma2_raw;
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
    double w = (double)tau / denom;  // Initial weight
    const double step = 1.0 / denom; // Decrement per iteration

    double sum = 0.0;
    for (int i = 0; i < tau; i++)
    {
        double xi = (r2[i] > 0.0) ? r2[i] : 0.0;
        sum = fma(w, xi, sum);
        w -= step;
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
 * Scalar gradient with backcast - optimized with STOPGRAD support
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
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        grad->d_omega = grad->d_alpha = grad->d_beta = 0.0;
        return;
    }

    const int prefetch_elems = 64;
    const bool track = cfg->track_clamps && stats;
    const bool sanitize = cfg->sanitize_r2;
    const bool stopgrad = (cfg->clamp_policy == GARCH_CLAMP_STOPGRAD);

    if (!track)
    {
        // FAST PATH (unchanged)
        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];

        bool clamped_0;
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast_detect(s, s, cfg, &clamped_0);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        if (stopgrad && clamped_0)
        {
            d_omega = d_alpha = d_beta = 0.0;
        }

        double dterm = d_nll_term_d_sigma2(r2_0, s);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_raw = omega + fma(alpha, r2_prev, beta * s_prev);
            const double level = (s_raw > s_prev) ? s_raw : s_prev;

            bool clamped_i;
            s = safe_sigma2_fast_detect(s_raw, level, cfg, &clamped_i);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2_prev);
            d_beta = fma(beta, d_beta, s_prev);

            if (stopgrad && clamped_i)
            {
                d_omega = 0.0;
                d_alpha = 0.0;
                d_beta = 0.0;
            }

            dterm = d_nll_term_d_sigma2(r2_curr, s);
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
        // TRACKING PATH - LOCAL COUNTER
        int local_clamps = 0; // Thread-local accumulator

        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];

        bool clamped_0;
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track_local_detect(s, s, cfg, &local_clamps, &clamped_0);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        if (stopgrad && clamped_0)
        {
            d_omega = d_alpha = d_beta = 0.0;
        }

        double dterm = d_nll_term_d_sigma2(r2_0, s);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_raw = omega + fma(alpha, r2_prev, beta * s_prev);
            const double level = (s_raw > s_prev) ? s_raw : s_prev;

            bool clamped_i;
            s = safe_sigma2_track_local_detect(s_raw, level, cfg, &local_clamps, &clamped_i);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2_prev);
            d_beta = fma(beta, d_beta, s_prev);

            if (stopgrad && clamped_i)
            {
                d_omega = 0.0;
                d_alpha = 0.0;
                d_beta = 0.0;
            }

            dterm = d_nll_term_d_sigma2(r2_curr, s);
            grad_omega = fma(dterm, d_omega, grad_omega);
            grad_alpha = fma(dterm, d_alpha, grad_alpha);
            grad_beta = fma(dterm, d_beta, grad_beta);

            s_prev = s;
        }

        // Batch flush: single atomic operation at the end
        atomic_fetch_add_explicit(&stats->sigma2_clamps, local_clamps, memory_order_relaxed);

        grad->d_omega = grad_omega;
        grad->d_alpha = grad_alpha;
        grad->d_beta = grad_beta;
    }
}

/**
 * Scalar NLL + gradient combined (most efficient for small n) with STOPGRAD support
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
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        grad->d_omega = grad->d_alpha = grad->d_beta = 0.0;
        return INFINITY;
    }

    const int prefetch_elems = 64;
    const bool track = cfg->track_clamps && stats;
    const bool sanitize = cfg->sanitize_r2;
    const bool stopgrad = (cfg->clamp_policy == GARCH_CLAMP_STOPGRAD);

    if (!track)
    {
        // FAST PATH (unchanged)
        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];

        bool clamped_0;
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast_detect(s, s, cfg, &clamped_0);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        if (stopgrad && clamped_0)
        {
            d_omega = d_alpha = d_beta = 0.0;
        }

        double dterm = inv * (1.0 - r2_0 * inv);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_raw = omega + fma(alpha, r2_prev, beta * s_prev);
            const double level = (s_raw > s_prev) ? s_raw : s_prev;

            bool clamped_i;
            s = safe_sigma2_fast_detect(s_raw, level, cfg, &clamped_i);

            inv = 1.0 / s;
            ln = log(s);
            nll += fma(r2_curr, inv, ln);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2_prev);
            d_beta = fma(beta, d_beta, s_prev);

            if (stopgrad && clamped_i)
            {
                d_omega = 0.0;
                d_alpha = 0.0;
                d_beta = 0.0;
            }

            dterm = inv * (1.0 - r2_curr * inv);
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
        // TRACKING PATH - LOCAL COUNTER
        int local_clamps = 0;

        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];

        bool clamped_0;
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track_local_detect(s, s, cfg, &local_clamps, &clamped_0);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        double d_omega = 1.0;
        double d_alpha = bc;
        double d_beta = bc;

        if (stopgrad && clamped_0)
        {
            d_omega = d_alpha = d_beta = 0.0;
        }

        double dterm = inv * (1.0 - r2_0 * inv);
        double grad_omega = dterm * d_omega;
        double grad_alpha = dterm * d_alpha;
        double grad_beta = dterm * d_beta;

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_raw = omega + fma(alpha, r2_prev, beta * s_prev);
            const double level = (s_raw > s_prev) ? s_raw : s_prev;

            bool clamped_i;
            s = safe_sigma2_track_local_detect(s_raw, level, cfg, &local_clamps, &clamped_i);

            inv = 1.0 / s;
            ln = log(s);
            nll += fma(r2_curr, inv, ln);

            d_omega = fma(beta, d_omega, 1.0);
            d_alpha = fma(beta, d_alpha, r2_prev);
            d_beta = fma(beta, d_beta, s_prev);

            if (stopgrad && clamped_i)
            {
                d_omega = 0.0;
                d_alpha = 0.0;
                d_beta = 0.0;
            }

            dterm = inv * (1.0 - r2_curr * inv);
            grad_omega = fma(dterm, d_omega, grad_omega);
            grad_alpha = fma(dterm, d_alpha, grad_alpha);
            grad_beta = fma(dterm, d_beta, grad_beta);

            s_prev = s;
        }

        // Batch flush
        atomic_fetch_add_explicit(&stats->sigma2_clamps, local_clamps, memory_order_relaxed);

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
 * AVX2 NLL + Gradient (vectorized Jacobian) - Optimized with STOPGRAD
 *
 * Key optimizations:
 * - Vectorize {d_omega, d_alpha, d_beta} in parallel using AVX2
 * - Reduce shuffle overhead with blend instructions
 * - Loop unswitching for tracking vs fast path
 * - STOPGRAD support: zero d_params when clamped
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
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        grad->d_omega = grad->d_alpha = grad->d_beta = 0.0;
        return INFINITY;
    }

    const int prefetch_elems = 64;
    const bool track = cfg->track_clamps && stats;
    const bool sanitize = cfg->sanitize_r2;
    const bool stopgrad = (cfg->clamp_policy == GARCH_CLAMP_STOPGRAD);

    // AVX2 constants
    const __m256d ZERO = _mm256_setzero_pd();
    const __m256d beta_vec = _mm256_set1_pd(beta);

    if (!track)
    {
        // ===== FAST PATH =====
        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];

        bool clamped_0;
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast_detect(s, s, cfg, &clamped_0);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        // Lane packing: [d_omega, d_alpha, d_beta, pad]
        __m256d d_params = _mm256_set_pd(0.0, bc, bc, 1.0);

        if (stopgrad && clamped_0)
        {
            d_params = ZERO;
        }

        double dterm = inv * (1.0 - r2_0 * inv);
        __m256d grad_vec = _mm256_mul_pd(_mm256_set1_pd(dterm), d_params);

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_raw = omega + fma(alpha, r2_prev, beta * s_prev);
            const double level = (s_raw > s_prev) ? s_raw : s_prev;

            bool clamped_i;
            double s_next = safe_sigma2_fast_detect(s_raw, level, cfg, &clamped_i);

            // NLL
            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2_curr, inv, ln);

            // Vectorized update: single _mm256_set_pd, no blends!
            // updates = [1.0, r2_prev, s_prev, 0.0]
            __m256d updates = _mm256_set_pd(0.0, s_prev, r2_prev, 1.0);
            d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

            if (stopgrad && clamped_i)
            {
                d_params = ZERO;
            }

            dterm = inv * (1.0 - r2_curr * inv);
            grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

            s_prev = s_next;
        }

        // Extract: lanes are [d_omega, d_alpha, d_beta, pad]
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
        int local_clamps = 0;

        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];

        bool clamped_0;
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_track_local_detect(s, s, cfg, &local_clamps, &clamped_0);

        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        __m256d d_params = _mm256_set_pd(0.0, bc, bc, 1.0);

        if (stopgrad && clamped_0)
        {
            d_params = ZERO;
        }

        double dterm = inv * (1.0 - r2_0 * inv);
        __m256d grad_vec = _mm256_mul_pd(_mm256_set1_pd(dterm), d_params);

        double s_prev = s;

        for (int i = 1; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];

            double s_raw = omega + fma(alpha, r2_prev, beta * s_prev);
            const double level = (s_raw > s_prev) ? s_raw : s_prev;

            bool clamped_i;
            double s_next = safe_sigma2_track_local_detect(s_raw, level, cfg, &local_clamps, &clamped_i);

            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2_curr, inv, ln);

            __m256d updates = _mm256_set_pd(0.0, s_prev, r2_prev, 1.0);
            d_params = _mm256_fmadd_pd(beta_vec, d_params, updates);

            if (stopgrad && clamped_i)
            {
                d_params = ZERO;
            }

            dterm = inv * (1.0 - r2_curr * inv);
            grad_vec = _mm256_fmadd_pd(_mm256_set1_pd(dterm), d_params, grad_vec);

            s_prev = s_next;
        }

        // Batch flush
        atomic_fetch_add_explicit(&stats->sigma2_clamps, local_clamps, memory_order_relaxed);

        double grad_array[4] __attribute__((aligned(32)));
        _mm256_store_pd(grad_array, grad_vec);

        grad->d_omega = grad_array[0];
        grad->d_alpha = grad_array[1];
        grad->d_beta = grad_array[2];

        return nll;
    }
}

/**
 * AVX2 NLL only (no gradient) - Optimized with sanitization support
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
    const double *restrict r2 = (const double *)__builtin_assume_aligned(data->r2, 32);
    const int n = data->n;
    const double bc = data->backcast;

    if (n <= 0 || !garch_valid_params(params))
    {
        return INFINITY;
    }

    const int prefetch_elems = 64;
    const bool track = cfg->track_clamps && stats;
    const bool sanitize = cfg->sanitize_r2;

    if (!track)
    {
        // FAST PATH
        double s = omega + (alpha + beta) * bc;
        s = safe_sigma2_fast(s, s, cfg);

        double r2_0 = sanitize ? clean_r2(r2[0]) : r2[0];
        double inv = 1.0 / s;
        double ln = log(s);
        double nll = fma(r2_0, inv, ln);

        // 4x unrolled
        int i = 1;
        for (; i < n - 3; i += 4)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_0 = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double s1 = omega + fma(alpha, r2_0, beta * s);
            const double l1 = (s1 > s) ? s1 : s;
            s1 = safe_sigma2_fast(s1, l1, cfg);
            inv = 1.0 / s1;
            ln = log(s1);
            double r2_1 = sanitize ? clean_r2(r2[i]) : r2[i];
            nll += fma(r2_1, inv, ln);

            double r2_2 = sanitize ? clean_r2(r2[i + 1]) : r2[i + 1];
            double s2 = omega + fma(alpha, r2_1, beta * s1);
            const double l2 = (s2 > s1) ? s2 : s1;
            s2 = safe_sigma2_fast(s2, l2, cfg);
            inv = 1.0 / s2;
            ln = log(s2);
            nll += fma(r2_2, inv, ln);

            double r2_3 = sanitize ? clean_r2(r2[i + 2]) : r2[i + 2];
            double s3 = omega + fma(alpha, r2_2, beta * s2);
            const double l3 = (s3 > s2) ? s3 : s2;
            s3 = safe_sigma2_fast(s3, l3, cfg);
            inv = 1.0 / s3;
            ln = log(s3);
            nll += fma(r2_3, inv, ln);

            double r2_4 = sanitize ? clean_r2(r2[i + 3]) : r2[i + 3];
            double s4 = omega + fma(alpha, r2_3, beta * s3);
            const double l4 = (s4 > s3) ? s4 : s3;
            s4 = safe_sigma2_fast(s4, l4, cfg);
            inv = 1.0 / s4;
            ln = log(s4);
            nll += fma(r2_4, inv, ln);

            s = s4;
        }

        // Remainder
        for (; i < n; i++)
        {
            if (i + prefetch_elems < n)
            {
                __builtin_prefetch(&r2[i + prefetch_elems], 0, 3);
            }

            double r2_prev = sanitize ? clean_r2(r2[i - 1]) : r2[i - 1];
            double r2_curr = sanitize ? clean_r2(r2[i]) : r2[i];
            double s_next = omega + fma(alpha, r2_prev, beta * s);
            const double level = (s_next > s) ? s_next : s;
            s_next = safe_sigma2_fast(s_next, level, cfg);
            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2_curr, inv, ln);
            s = s_next;
        }

        return nll;
    }
    else
    {
        // TRACKING PATH
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
            inv = 1.0 / s_next;
            ln = log(s_next);
            nll += fma(r2_curr, inv, ln);
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

    // First step: consistent with recursion (use max(s_raw, prev_sigma))
    double s0_raw = omega + fma(alpha, last_r2, beta * last_sigma2);
    const double level0 = (s0_raw > last_sigma2) ? s0_raw : last_sigma2;
    double s0 = safe_sigma2_fast(s0_raw, level0, cfg);
    sigma2[0] = s0;

    double z0 = isfinite(shocks[0]) ? shocks[0] : 0.0;
    returns[0] = z0 * sqrt(s0);

    // Subsequent steps: same policy throughout
    for (int i = 1; i < n; i++)
    {
        double zi = isfinite(shocks[i]) ? shocks[i] : 0.0;
        double r_prev_sq = returns[i - 1] * returns[i - 1];
        double s_raw = omega + fma(alpha, r_prev_sq, beta * sigma2[i - 1]);
        const double level = (s_raw > sigma2[i - 1]) ? s_raw : sigma2[i - 1];
        double s = safe_sigma2_fast(s_raw, level, cfg);
        sigma2[i] = s;
        returns[i] = zi * sqrt(s);
    }
}