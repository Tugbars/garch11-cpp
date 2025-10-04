#pragma once
//inspired from https://github.com/kkew3/garch11/blob/main/src/garch11.cpp
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
namespace garch11 {

/**
 * @brief Configuration structure for GARCH computations.
 * 
 * Allows tuning of numerical stability parameters and clamping behavior.
 * eps_base is scaled adaptively by dynamic_eps() based on the magnitude of sigma2.
 * eps_min provides a hard floor to prevent underflow to subnormal numbers.
 * track_clamps enables diagnostic counting of clamping events.
 */
struct GarchConfig {
    /// Base epsilon for adaptive scaling (machine epsilon by default, ~2e-16).
    double eps_base = std::numeric_limits<double>::epsilon();
    /// Minimum epsilon floor to avoid subnormal arithmetic; adjustable for precision needs.
    double eps_min = 1e-16;
    /// If true, increment ClampStats counters when clamping occurs for diagnostics.
    bool track_clamps = true;
};

/**
 * @brief Statistics for tracking clamping events during computations.
 * 
 * Used to monitor numerical stability issues, e.g., how often sigma2 was floored to epsilon.
 */
struct ClampStats {
    /// Number of times sigma2 was clamped to the minimum epsilon.
    int sigma2_clamps = 0;
};

/**
 * @brief Policy for handling invalid inputs (e.g., non-stationary parameters or n <= 0).
 * 
 * ReturnInf: Returns +inf for NLL or zeros for Jacobian (safe for optimizers).
 * Throw: Throws std::invalid_argument for strict error handling.
 */
enum class InputPolicy { ReturnInf, Throw };

/**
 * @brief Policy for handling invalid parameters in prediction.
 * 
 * Clamp: Silently returns without updating outputs (safe for simulation loops).
 * Throw: Throws std::invalid_argument for strict error handling.
 */
enum class PredictPolicy{ Clamp, Throw };

/**
 * @brief Sanitizes the EWMA decay parameter lambda to ensure it's in (0,1).
 * 
 * Clamps invalid values (e.g., <=0 or >=1) to a sensible default (0.97).
 * This prevents degenerate weighting (no decay or negative weights) in EWMA backcast.
 * 
 * @param lambda Raw lambda value.
 * @return Sanitized lambda in (0,1), defaulting to 0.97 if invalid.
 */
inline double sanitize_lambda(double lambda) {
    // Clamp to sensible default if out of (0,1)
    return (lambda > 0.0 && lambda < 1.0) ? lambda : 0.97;
}

// -----------------------------
// Helpers
// -----------------------------

/**
 * @brief Validates GARCH(1,1) parameters for positivity and stationarity.
 * 
 * Ensures omega > 0 (positive long-run variance), alpha >= 0, beta >= 0,
 * and alpha + beta < 1 (stationarity condition to ensure finite unconditional variance).
 * 
 * @param omega Constant term in variance equation.
 * @param alpha ARCH parameter (impact of lagged squared returns).
 * @param beta GARCH parameter (persistence of lagged variance).
 * @return true if parameters are valid, false otherwise.
 */
inline bool valid_params(double omega, double alpha, double beta) {
    return (omega > 0.0) && (alpha >= 0.0) && (beta >= 0.0) && ((alpha + beta) < 1.0);
}

/**
 * @brief Checks if a value is finite (not NaN or Inf).
 * 
 * Uses std::isfinite for portable floating-point checks.
 * 
 * @param x Value to check.
 * @return true if finite, false otherwise.
 */
inline bool is_finite(double x) { return std::isfinite(x); }

/**
 * @brief Computes an adaptive epsilon based on the scale of the variance level.
 * 
 * Scales eps_base by max(1, level_for_scale) to maintain relative precision
 * (e.g., larger eps for high-volatility series). Floors at eps_min to avoid subnormals.
 * This prevents numerical instability in log() and division without over-clamping.
 * 
 * @param level_for_scale Typical sigma2 magnitude for adaptive scaling.
 * @param cfg Configuration with eps_base and eps_min.
 * @return Adaptive epsilon value.
 */
inline double dynamic_eps(double level_for_scale, const GarchConfig& cfg) {
    double scale = std::max(1.0, level_for_scale); // Avoid scaling below 1
    double e = cfg.eps_base * scale;
    return (e < cfg.eps_min) ? cfg.eps_min : e;
}

/**
 * @brief Safely clamps raw sigma2 to a minimum epsilon if too small.
 * 
 * Prevents log(0) or division-by-zero in NLL terms. Uses dynamic_eps for scale-adaptive clamping.
 * Increments stats counter if track_clamps is enabled.
 * 
 * @param sigma2_raw Unclamped variance.
 * @param level_for_scale Scale for adaptive epsilon (often sigma2 itself).
 * @param cfg Configuration for eps values.
 * @param stats Optional pointer to ClampStats for diagnostics.
 * @return Clamped sigma2 (sigma2_raw if >= eps, else eps).
 */
inline double safe_sigma2(double sigma2_raw, double level_for_scale,
                          const GarchConfig& cfg, ClampStats* stats) {
    double eps = dynamic_eps(level_for_scale, cfg);
    if (sigma2_raw < eps) {
        if (stats && cfg.track_clamps) ++stats->sigma2_clamps; // Track for diagnostics
        return eps;
    }
    return sigma2_raw;
}

/**
 * @brief Computes a single scaled NLL term: log(sigma2) + r2 / sigma2.
 * 
 * Applies safe_sigma2 clamping to sigma2_raw before computation.
 * Uses level_for_scale for adaptive eps in clamping.
 * 
 * @param r2_i Squared return (innovation variance) at time i.
 * @param sigma2_raw Raw conditional variance at time i.
 * @param level_for_scale Scale for adaptive clamping (often sigma2).
 * @param cfg Configuration for eps and clamping.
 * @param stats Optional ClampStats pointer.
 * @return Scaled NLL term (clamped if necessary).
 */
inline double nll_term_scaled(double r2_i, double sigma2_raw, double level_for_scale,
                              const GarchConfig& cfg, ClampStats* stats) {
    double s = safe_sigma2(sigma2_raw, level_for_scale, cfg, stats);
    double inv = 1.0 / s;
    return std::log(s) + r2_i * inv; // Omits constant terms (e.g., log(2pi)/2) for optimization
}

/**
 * @brief Computes the derivative of the NLL term w.r.t. sigma2: (1 - r2 / sigma2) / sigma2.
 * 
 * Used in analytical Jacobian. Applies safe_sigma2 clamping.
 * 
 * @param r2_i Squared return at time i.
 * @param sigma2_raw Raw conditional variance.
 * @param level_for_scale Scale for clamping.
 * @param cfg Configuration.
 * @param stats Optional ClampStats.
 * @return d(NLL_term)/d_sigma2, evaluated at clamped sigma2.
 */
inline double d_term_d_sigma2_scaled(double r2_i, double sigma2_raw, double level_for_scale,
                                     const GarchConfig& cfg, ClampStats* stats) {
    double s = safe_sigma2(sigma2_raw, level_for_scale, cfg, stats);
    double inv = 1.0 / s;
    return inv * (1.0 - r2_i * inv); // Chain rule: partial of log(s) + r2/s w.r.t. s
}

// -----------------------------
// Core routines (policy-enabled)
// -----------------------------

/**
 * @brief Backcast methods for pre-sample variance initialization.
 * 
 * Mean: Simple average; robust for small samples.
 * TriangularEarly: Linear weights decreasing from earliest obs (prioritizes t=1 era).
 * EWMA: Exponential decay (lambda ~0.94-0.99); smooth emphasis on early obs.
 */
enum class BackcastMethod { Mean, TriangularEarly, EWMA };

/**
 * @brief Computes backcast variance using simple mean of squared returns.
 * 
 * Treats all observations equally; stable for short series but ignores temporal structure.
 * Clamps r2[i] >=0 to ensure non-negative variance.
 * 
 * @param r2 Array of squared returns [r2_1, ..., r2_n].
 * @param n Length of r2.
 * @return Mean of max(r2[i], 0), or 0 if n<=0.
 */
inline double calc_backcast_mean(const double *r2, int n) {
    if (n <= 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::max(r2[i], 0.0); // Ensure non-negative
    return s / static_cast<double>(n);
}

/**
 * @brief Computes backcast variance using EWMA with decay lambda.
 * 
 * Weights decrease exponentially from r2[0] (earliest gets highest weight).
 * Uses tau=min(n,75) for computational efficiency (common in GARCH software).
 * Falls back to mean if weights underflow or become invalid.
 * 
 * @param r2 Array of squared returns.
 * @param n Length of r2.
 * @param lambda Decay factor in (0,1); higher values spread weight more evenly.
 * @return Weighted average, or fallback mean if unstable; 0 if n<=0.
 */
inline double calc_backcast_ewma(const double *r2, int n, double lambda = 0.97) {
    lambda = sanitize_lambda(lambda); // Sanitize to prevent invalid weights
    int tau = (n < 75) ? n : 75; // Limit tau for efficiency; 75 is empirical cutoff
    if (tau <= 1) return (tau == 1) ? std::max(r2[0], 0.0) : 0.0;
    double wsum = 0.0, sum = 0.0, w = 1.0;
    for (int i = 0; i < tau; ++i) {
        // earlier obs (i=0) get largest weight
        sum += w * std::max(r2[i], 0.0);
        wsum += w;
        w *= lambda; // Decay toward later obs
    }
    // Stability guard: if something went off, fall back to mean over the same tau
    if (!(wsum > 0.0) || !std::isfinite(wsum) || !std::isfinite(sum)) { // Check sum finiteness too
        double s = 0.0;
        for (int i = 0; i < tau; ++i) s += std::max(r2[i], 0.0);
        return (tau > 0) ? s / static_cast<double>(tau) : 0.0;
    }
    double bc = sum / wsum;
    return (bc > 0.0) ? bc : 0.0;
}

/**
 * @brief Computes pre-sample variance (backcast) for GARCH initialization.
 * 
 * Approximates sigma2_0 or epsilon2_0 using sample r2[0..n-1].
 * Defaults to TriangularEarly (linear decay prioritizing early obs).
 * 
 * @param r2 Array of squared returns.
 * @param n Length of r2.
 * @param m Backcast method (Mean, TriangularEarly, EWMA).
 * @param lambda Decay for EWMA only.
 * @return Non-negative backcast variance estimate; 0 if n<=0.
 */
inline double calc_backcast(const double* r2, int n,
                            BackcastMethod m = BackcastMethod::TriangularEarly,
                            double lambda = 0.97) {
    switch (m) {
        case BackcastMethod::Mean:
            return calc_backcast_mean(r2, n); // Reuse for consistency
        case BackcastMethod::EWMA:
            return calc_backcast_ewma(r2, n, lambda);
        case BackcastMethod::TriangularEarly:
        default: {
            int tau = (n < 75) ? n : 75; // Limit for efficiency
            if (tau <= 1) return (tau == 1) ? std::max(r2[0], 0.0) : 0.0;
            const double denom = 0.5 * tau * (tau + 1); // Sum of weights 1+2+...+tau = tau(tau+1)/2
            double sum = 0.0;
            for (int i = 0; i < tau; ++i) {
                const double w = static_cast<double>(tau - i) / denom; // largest at i=0 (early emphasis)
                sum += std::max(r2[i], 0.0) * w;
            }
            return (sum > 0.0) ? sum : 0.0;
        }
    }
}

/**
 * @brief Computes the negative log-likelihood (NLL) without backcast initialization.
 * 
 * Initializes sigma2_0 = omega / (1 - alpha - beta) (unconditional variance, assuming stationarity).
 * Recurses sigma2_t = omega + alpha * r2_{t-1} + beta * sigma2_{t-1}.
 * NLL = sum [log(sigma2_t) + r2_t / sigma2_t] (scaled; constants omitted).
 * 
 * @tparam policy Input handling (ReturnInf or Throw).
 * @param x Parameter vector [omega, alpha, beta].
 * @param r2 Squared returns [r2_1, ..., r2_n].
 * @param n Number of observations.
 * @param cfg Numerical configuration.
 * @param stats Optional clamping diagnostics.
 * @return NLL value; +inf or throw on invalid input.
 */
template <InputPolicy policy = InputPolicy::ReturnInf>
inline double calc_fun_policy(const double *x, const double *r2, int n,
                              const GarchConfig& cfg = {}, ClampStats* stats = nullptr) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == InputPolicy::Throw) throw std::invalid_argument("calc_fun: bad input");
        return std::numeric_limits<double>::infinity();
    }
    const double one_over_one_minus_ab = 1.0 / (1.0 - alpha - beta); // Unconditional variance factor
    double sigma2 = omega * one_over_one_minus_ab; // Initial sigma2_0 (stationary assumption)
    double y = nll_term_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    for (int i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2; // GARCH recursion
        y += nll_term_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
    }
    return y;
}

/**
 * @brief Computes the Jacobian (gradient) of NLL w.r.t. parameters without backcast.
 * 
 * Analytical derivatives via chain rule: accumulates partial sigma2_t / partial {omega,alpha,beta}
 * multiplied by d(NLL_term)/d_sigma2_t. Initializes from unconditional variance.
 * 
 * @tparam policy Input handling.
 * @param x Parameters [omega, alpha, beta].
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Output [dNLL/domega, dNLL/dalpha, dNLL/dbeta].
 * @param cfg Configuration.
 * @param stats Diagnostics.
 */
template <InputPolicy policy = InputPolicy::ReturnInf>
inline void calc_jac_policy(const double *x, const double *r2, int n, double *out_jac,
                            const GarchConfig& cfg = {}, ClampStats* stats = nullptr) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == InputPolicy::Throw) throw std::invalid_argument("calc_jac: bad input");
        out_jac[0] = out_jac[1] = out_jac[2] = 0.0;
        return;
    }
    const double one_over_one_minus_ab = 1.0 / (1.0 - alpha - beta);
    double sigma2 = omega * one_over_one_minus_ab;
    // Initial partials w.r.t. unconditional sigma2_0
    double d_sigma2_d_omega = one_over_one_minus_ab;
    double d_sigma2_d_alpha = omega * one_over_one_minus_ab * one_over_one_minus_ab;
    double d_sigma2_d_beta = d_sigma2_d_alpha; // Symmetric in alpha/beta for init
    double dterm = d_term_d_sigma2_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    double par_y_omega = dterm * d_sigma2_d_omega;
    double par_y_alpha = dterm * d_sigma2_d_alpha;
    double par_y_beta = dterm * d_sigma2_d_beta;
    double sigma2_prev = sigma2;
    for (int i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        // Propagate partials via chain rule: d_sigma2_t / d_theta = 1 + beta * d_sigma2_{t-1} / d_theta (for omega)
        // Similar for alpha (r2_{t-1} + beta * prev) and beta (sigma2_{t-1} + beta * prev)
        d_sigma2_d_omega = 1.0 + beta * d_sigma2_d_omega;
        d_sigma2_d_alpha = r2[i - 1] + beta * d_sigma2_d_alpha;
        d_sigma2_d_beta = sigma2_prev + beta * d_sigma2_d_beta;
        dterm = d_term_d_sigma2_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
        par_y_omega += dterm * d_sigma2_d_omega;
        par_y_alpha += dterm * d_sigma2_d_alpha;
        par_y_beta += dterm * d_sigma2_d_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
}

/**
 * @brief Computes NLL and Jacobian together without backcast (combined for efficiency).
 * 
 * Identical to separate calls but avoids redundant sigma2 recursion.
 * 
 * @tparam policy Input handling.
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient output.
 * @param cfg Configuration.
 * @param stats Diagnostics.
 * @return NLL value; +inf or throw on invalid.
 */
template <InputPolicy policy = InputPolicy::ReturnInf>
inline double calc_fun_jac_policy(const double *x, const double *r2, int n, double *out_jac,
                                  const GarchConfig& cfg = {}, ClampStats* stats = nullptr) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == InputPolicy::Throw) throw std::invalid_argument("calc_fun_jac: bad input");
        out_jac[0] = out_jac[1] = out_jac[2] = 0.0;
        return std::numeric_limits<double>::infinity();
    }
    const double one_over_one_minus_ab = 1.0 / (1.0 - alpha - beta);
    double sigma2 = omega * one_over_one_minus_ab;
    double y = nll_term_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    double d_sigma2_d_omega = one_over_one_minus_ab;
    double d_sigma2_d_alpha = omega * one_over_one_minus_ab * one_over_one_minus_ab;
    double d_sigma2_d_beta = d_sigma2_d_alpha;
    double dterm = d_term_d_sigma2_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    double par_y_omega = dterm * d_sigma2_d_omega;
    double par_y_alpha = dterm * d_sigma2_d_alpha;
    double par_y_beta = dterm * d_sigma2_d_beta;
    double sigma2_prev = sigma2;
    for (int i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        y += nll_term_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats); // Accumulate NLL
        d_sigma2_d_omega = 1.0 + beta * d_sigma2_d_omega;
        d_sigma2_d_alpha = r2[i - 1] + beta * d_sigma2_d_alpha;
        d_sigma2_d_beta = sigma2_prev + beta * d_sigma2_d_beta;
        dterm = d_term_d_sigma2_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
        par_y_omega += dterm * d_sigma2_d_omega;
        par_y_alpha += dterm * d_sigma2_d_alpha;
        par_y_beta += dterm * d_sigma2_d_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
    return y;
}

/**
 * @brief NLL with backcast initialization for pre-sample variance.
 * 
 * Initializes sigma2_0 = omega + (alpha + beta) * bc, where bc approximates pre-sample variance.
 * Better for finite samples than unconditional init.
 * 
 * @tparam policy Input handling.
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param cfg Configuration.
 * @param stats Diagnostics.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 * @return NLL; +inf or throw on invalid.
 */
template <InputPolicy policy = InputPolicy::ReturnInf>
inline double calc_fun_bc_policy(const double *x, const double *r2, int n,
                                 const GarchConfig& cfg = {}, ClampStats* stats = nullptr,
                                 BackcastMethod m = BackcastMethod::TriangularEarly,
                                 double lambda = 0.97) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == InputPolicy::Throw) throw std::invalid_argument("calc_fun_bc: bad input");
        return std::numeric_limits<double>::infinity();
    }
    double bc = calc_backcast(r2, n, m, lambda); // Data-driven pre-sample variance
    double sigma2 = omega + (alpha + beta) * bc; // Backcast init (approximates infinite AR(1) sum)
    double y = nll_term_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    for (int i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2;
        y += nll_term_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
    }
    return y;
}

/**
 * @brief Jacobian of NLL with backcast initialization.
 * 
 * Initial partials: d_sigma2_0/d_omega=1, d/d_alpha=bc, d/d_beta=bc (bc independent of params).
 * 
 * @tparam policy Input handling.
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient [dNLL/domega, ...].
 * @param cfg Configuration.
 * @param stats Diagnostics.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 */
template <InputPolicy policy = InputPolicy::ReturnInf>
inline void calc_jac_bc_policy(const double *x, const double *r2, int n, double *out_jac,
                               const GarchConfig& cfg = {}, ClampStats* stats = nullptr,
                               BackcastMethod m = BackcastMethod::TriangularEarly,
                               double lambda = 0.97) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == InputPolicy::Throw) throw std::invalid_argument("calc_jac_bc: bad input");
        out_jac[0] = out_jac[1] = out_jac[2] = 0.0;
        return;
    }
    double bc = calc_backcast(r2, n, m, lambda);
    double sigma2 = omega + (alpha + beta) * bc;
    // Initial partials for backcast init (bc treated as constant w.r.t. params)
    double d_sigma2_d_omega = 1.0;
    double d_sigma2_d_alpha = bc;
    double d_sigma2_d_beta = bc;
    double dterm = d_term_d_sigma2_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    double par_y_omega = dterm * d_sigma2_d_omega;
    double par_y_alpha = dterm * d_sigma2_d_alpha;
    double par_y_beta = dterm * d_sigma2_d_beta;
    double sigma2_prev = sigma2;
    for (int i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        d_sigma2_d_omega = 1.0 + beta * d_sigma2_d_omega;
        d_sigma2_d_alpha = r2[i - 1] + beta * d_sigma2_d_alpha;
        d_sigma2_d_beta = sigma2_prev + beta * d_sigma2_d_beta;
        dterm = d_term_d_sigma2_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
        par_y_omega += dterm * d_sigma2_d_omega;
        par_y_alpha += dterm * d_sigma2_d_alpha;
        par_y_beta += dterm * d_sigma2_d_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
}

/**
 * @brief NLL and Jacobian with backcast (combined for efficiency).
 * 
 * @tparam policy Input handling.
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient.
 * @param cfg Configuration.
 * @param stats Diagnostics.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 * @return NLL; +inf or throw on invalid.
 */
template <InputPolicy policy = InputPolicy::ReturnInf>
inline double calc_fun_jac_bc_policy(const double *x, const double *r2, int n, double *out_jac,
                                     const GarchConfig& cfg = {}, ClampStats* stats = nullptr,
                                     BackcastMethod m = BackcastMethod::TriangularEarly,
                                     double lambda = 0.97) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == InputPolicy::Throw) throw std::invalid_argument("calc_fun_jac_bc: bad input");
        out_jac[0] = out_jac[1] = out_jac[2] = 0.0;
        return std::numeric_limits<double>::infinity();
    }
    double bc = calc_backcast(r2, n, m, lambda);
    double sigma2 = omega + (alpha + beta) * bc;
    double y = nll_term_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    double d_sigma2_d_omega = 1.0;
    double d_sigma2_d_alpha = bc;
    double d_sigma2_d_beta = bc;
    double dterm = d_term_d_sigma2_scaled(r2[0], sigma2, /*level*/sigma2, cfg, stats);
    double par_y_omega = dterm * d_sigma2_d_omega;
    double par_y_alpha = dterm * d_sigma2_d_alpha;
    double par_y_beta = dterm * d_sigma2_d_beta;
    double sigma2_prev = sigma2;
    for (int i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        y += nll_term_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
        d_sigma2_d_omega = 1.0 + beta * d_sigma2_d_omega;
        d_sigma2_d_alpha = r2[i - 1] + beta * d_sigma2_d_alpha;
        d_sigma2_d_beta = sigma2_prev + beta * d_sigma2_d_beta;
        dterm = d_term_d_sigma2_scaled(r2[i], sigma2, /*level*/sigma2, cfg, stats);
        par_y_omega += dterm * d_sigma2_d_omega;
        par_y_alpha += dterm * d_sigma2_d_alpha;
        par_y_beta += dterm * d_sigma2_d_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
    return y;
}

/**
 * @brief Transforms parameters to conditional variance path (no backcast).
 * 
 * Initializes sigma2_0 from unconditional variance; recurses forward.
 * Clamps each sigma2_t for stability; uses max(s_raw, prev) for adaptive scaling.
 * 
 * @param x Parameters [omega, alpha, beta].
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_sigma2 Output variance path [sigma2_1, ..., sigma2_n].
 * @param cfg Configuration.
 * @param stats Diagnostics.
 */
inline void transform(const double *x, const double *r2, int n, double *out_sigma2,
                      const GarchConfig& cfg = {}, ClampStats* stats = nullptr) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) return; // Silent; caller checks out_sigma2
    const double one_over_one_minus_ab = 1.0 / (1.0 - alpha - beta);
    double s0_raw = omega * one_over_one_minus_ab;
    out_sigma2[0] = safe_sigma2(s0_raw, /*level*/s0_raw, cfg, stats);
    for (int i = 1; i < n; ++i) {
        double s_raw = omega + alpha * r2[i - 1] + beta * out_sigma2[i - 1];
        out_sigma2[i] = safe_sigma2(s_raw, /*level*/std::max(s_raw, out_sigma2[i - 1]), cfg, stats);
        // max(s_raw, prev) adapts to series scale while preventing sudden drops
    }
}

/**
 * @brief Variance path with backcast initialization.
 * 
 * Uses backcast for sigma2_0; otherwise same as transform().
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_sigma2 Output path.
 * @param cfg Configuration.
 * @param stats Diagnostics.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 */
inline void transform_bc(const double *x, const double *r2, int n, double *out_sigma2,
                         const GarchConfig& cfg = {}, ClampStats* stats = nullptr,
                         BackcastMethod m = BackcastMethod::TriangularEarly,
                         double lambda = 0.97) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        // For n<=0, leave out_sigma2 unchanged (caller must handle)
        return;
    }
    double bc = calc_backcast(r2, n, m, lambda);
    double s0_raw = omega + (alpha + beta) * bc;
    out_sigma2[0] = safe_sigma2(s0_raw, /*level*/s0_raw, cfg, stats);
    for (int i = 1; i < n; ++i) {
        double s_raw = omega + alpha * r2[i - 1] + beta * out_sigma2[i - 1];
        out_sigma2[i] = safe_sigma2(s_raw, /*level*/std::max(s_raw, out_sigma2[i - 1]), cfg, stats);
    }
}

/**
 * @brief Simulates future GARCH paths (forecasting/prediction).
 * 
 * Generates sigma2_t and r_t = z_t * sqrt(sigma2_t) for n steps, starting from last observed.
 * out_r2 holds raw innovations r_t (not squared); z_t from randn_nums (standard normal).
 * Clamps invalid last_sigma2/last_r2 to 0; uses adaptive scaling.
 * 
 * @tparam policy Input handling for params.
 * @param x Parameters.
 * @param last_sigma2 Last observed conditional variance.
 * @param last_r2 Last observed squared return (r_{t-1}^2).
 * @param randn_nums Standard normal shocks [z_1, ..., z_n].
 * @param n Forecast horizon.
 * @param out_sigma2 Output variances.
 * @param out_r2 Output innovations r_t.
 * @param cfg Configuration.
 * @param stats Diagnostics.
 */
template <PredictPolicy policy = PredictPolicy::Clamp>
inline void predict(const double *x, double last_sigma2, double last_r2,
                    const double *randn_nums, int n, double *out_sigma2, double *out_r2,
                    const GarchConfig& cfg = {}, ClampStats* stats = nullptr) {
    const double omega = x[0], alpha = x[1], beta = x[2];
    if (n <= 0 || !valid_params(omega, alpha, beta)) {
        if constexpr (policy == PredictPolicy::Throw) throw std::invalid_argument("predict: bad input");
        return; // Silent for Clamp policy
    }
    if (!is_finite(last_sigma2) || last_sigma2 < 0.0) last_sigma2 = 0.0; // Clamp invalid to non-negative
    if (!is_finite(last_r2)) last_r2 = 0.0; // Avoid NaN propagation
    // t = 0 (first forecast step)
    double s0_raw = omega + alpha * last_r2 + beta * last_sigma2;
    double s0 = safe_sigma2(s0_raw, /*level*/std::max(omega, s0_raw), cfg, stats); // Scale by omega if s0 small
    out_sigma2[0] = s0;
    double z0 = is_finite(randn_nums[0]) ? randn_nums[0] : 0.0; // Default z=0 if invalid
    out_r2[0] = z0 * std::sqrt(s0); // r_t = z_t * sqrt(sigma2_t)
    for (int i = 1; i < n; ++i) {
        double zi = is_finite(randn_nums[i]) ? randn_nums[i] : 0.0;
        double s_raw = omega + alpha * (out_r2[i - 1] * out_r2[i - 1]) + beta * out_sigma2[i - 1]; // Use realized r^2
        double s = safe_sigma2(s_raw, /*level*/std::max(s_raw, out_sigma2[i - 1]), cfg, stats);
        out_sigma2[i] = s;
        out_r2[i] = zi * std::sqrt(s);
    }
}

/**
 * @brief Computes NLL given pre-computed sigma2 path.
 * 
 * For use after transform() or external variance paths.
 * Applies safe_sigma2 to each sigma2[i] for consistency.
 * 
 * @param r2 Squared returns.
 * @param n Observations.
 * @param sigma2 Conditional variances [sigma2_1, ..., sigma2_n].
 * @param cfg Configuration.
 * @param stats Diagnostics.
 * @return NLL sum.
 */
inline double nll(const double *r2, int n, const double *sigma2,
                  const GarchConfig& cfg = {}, ClampStats* stats = nullptr) {
    double y = 0.0;
    for (int i = 0; i < n; ++i) {
        // Use sigma2[i] itself as the level for scaling; caller provides sigma2 path.
        double s = safe_sigma2(sigma2[i], /*level*/sigma2[i], cfg, stats);
        y += std::log(s) + r2[i] / s;
    }
    return y;
}

// -----------------------------
// Backward-compatible wrappers (original signatures)
// -----------------------------

/**
 * @brief NLL without backcast (ReturnInf policy).
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @return NLL.
 */
inline double calc_fun(const double *x, const double *r2, int n) {
    return calc_fun_policy<InputPolicy::ReturnInf>(x, r2, n);
}

/**
 * @brief Jacobian without backcast (ReturnInf policy).
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient.
 */
inline void calc_jac(const double *x, const double *r2, int n, double *out_jac) {
    calc_jac_policy<InputPolicy::ReturnInf>(x, r2, n, out_jac);
}

/**
 * @brief NLL + Jacobian without backcast.
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient.
 * @return NLL.
 */
inline double calc_fun_jac(const double *x, const double *r2, int n, double *out_jac) {
    return calc_fun_jac_policy<InputPolicy::ReturnInf>(x, r2, n, out_jac);
}

/**
 * @brief NLL with backcast (default TriangularEarly, lambda=0.97).
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 * @return NLL.
 */
inline double calc_fun_bc(const double *x, const double *r2, int n,
                          BackcastMethod m, double lambda = 0.97) {
    return calc_fun_bc_policy<InputPolicy::ReturnInf>(x, r2, n, GarchConfig{}, nullptr, m, lambda);
}

/**
 * @brief Jacobian with backcast (default TriangularEarly).
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 */
inline void calc_jac_bc(const double *x, const double *r2, int n, double *out_jac,
                          BackcastMethod m, double lambda = 0.97) {
    calc_jac_bc_policy<InputPolicy::ReturnInf>(x, r2, n, out_jac, GarchConfig{}, nullptr, m, lambda);
}

/**
 * @brief NLL + Jacobian with backcast.
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_jac Gradient.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 * @return NLL.
 */
inline double calc_fun_jac_bc(const double *x, const double *r2, int n, double *out_jac,
                              BackcastMethod m, double lambda = 0.97) {
    return calc_fun_jac_bc_policy<InputPolicy::ReturnInf>(x, r2, n, out_jac, GarchConfig{}, nullptr, m, lambda);
}

/**
 * @brief Variance path without backcast (defaults).
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_sigma2 Path.
 */
inline void transform(const double *x, const double *r2, int n, double *out_sigma2) {
    transform(x, r2, n, out_sigma2, GarchConfig{}, nullptr);
}

/**
 * @brief Variance path with backcast (default TriangularEarly).
 * 
 * @param x Parameters.
 * @param r2 Squared returns.
 * @param n Observations.
 * @param out_sigma2 Path.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 */
inline void transform_bc(const double *x, const double *r2, int n, double *out_sigma2,
                         BackcastMethod m, double lambda = 0.97) {
    transform_bc(x, r2, n, out_sigma2, GarchConfig{}, nullptr, m, lambda);
}

/**
 * @brief Prediction simulation (Clamp policy).
 * 
 * @param x Parameters.
 * @param last_sigma2 Last variance.
 * @param last_r2 Last squared return.
 * @param randn_nums Shocks.
 * @param n Horizon.
 * @param out_sigma2 Variances.
 * @param out_r2 Innovations.
 */
inline void predict(const double *x, double last_sigma2, double last_r2,
                    const double *randn_nums, int n, double *out_sigma2, double *out_r2) {
    predict<PredictPolicy::Clamp>(x, last_sigma2, last_r2, randn_nums, n, out_sigma2, out_r2);
}

/**
 * @brief NLL from pre-computed path (defaults).
 * 
 * @param r2 Squared returns.
 * @param n Observations.
 * @param sigma2 Variances.
 * @return NLL.
 */
inline double nll(const double *r2, int n, const double *sigma2) {
    return nll(r2, n, sigma2, GarchConfig{}, nullptr);
}

} // namespace garch11
