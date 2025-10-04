#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include "garch11.hpp"
#include "neldermead.hpp"

namespace garch11 {

/**
 * @brief Wrapper for GARCH(1,1) negative log-likelihood.
 * 
 * @param params Parameters [omega, alpha, beta].
 * @param r2 Squared returns.
 * @param n Number of observations.
 * @param cfg GARCH configuration.
 * @param stats Clamping diagnostics.
 * @param m Backcast method.
 * @param lambda EWMA decay.
 * @return NLL value.
 */
double LogLikelihood(const std::vector<double>& params, const double* r2, int n,
                     const GarchConfig& cfg, ClampStats* stats,
                     BackcastMethod m, double lambda) {
    const double omega = params[0], alpha = params[1], beta = params[2];
    if (omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 1.0) {
        return 1e10; // Large penalty for invalid parameters
    }
    double grad[3];
    return calc_fun_jac_bc_policy<InputPolicy::ReturnInf>(
        params.data(), r2, n, grad, cfg, stats, m, lambda);
}

/**
 * @brief Generates synthetic GARCH(1,1) data for testing.
 * 
 * @param params True parameters [omega, alpha, beta].
 * @param n Number of observations.
 * @param seed Random seed.
 * @param out_r2 Output squared returns.
 * @param out_sigma2 Output conditional variances.
 */
void GenerateGarchData(const double* params, int n, unsigned int seed,
                       double* out_r2, double* out_sigma2) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    const double omega = params[0], alpha = params[1], beta = params[2];
    
    // Initialize with unconditional variance
    double sigma2 = omega / (1.0 - alpha - beta);
    for (int i = 0; i < n; ++i) {
        double z = dist(rng);
        double r = z * std::sqrt(sigma2);
        out_r2[i] = r * r; // Squared returns
        out_sigma2[i] = sigma2;
        sigma2 = omega + alpha * out_r2[i] + beta * sigma2; // Next variance
    }
}

} // namespace garch11

int main() {
    // Parameters for testing
    const int N = 3; // GARCH(1,1) parameters: omega, alpha, beta
    const double num_iters = 1; // Starting iteration (unused in NelderMead)
    const double max_iters = 500; // Increased from 1e2
    const double tolerance = 1e-15; // Original stricter tolerance
    const int n_data = 1000; // Number of data points
    const unsigned int seed = 42; // For reproducibility

    // Generate synthetic GARCH data
    std::vector<double> true_params = {0.00002, 0.1, 0.85}; // omega, alpha, beta
    std::vector<double> r2(n_data), sigma2(n_data);
    garch11::GenerateGarchData(true_params.data(), n_data, seed, r2.data(), sigma2.data());

    // Initialize simplex (from your example)
    std::vector<std::vector<double>> s(N, std::vector<double>(N + 1));
    s[0] = {0.00002, 0.000018, 0.000022, 0.000025}; // omega, tighter around 0.00002
    s[1] = {0.10, 0.095, 0.105, 0.11};             // alpha, tighter around 0.1
    s[2] = {0.85, 0.84, 0.86, 0.845};              // beta, tighter around 0.85

    // Wrap LogLikelihood with r2 and other params
    garch11::GarchConfig cfg; // Default config
    garch11::ClampStats stats;
    std::function<double(const std::vector<double>&)> obj_func =
    [&](const std::vector<double>& params) -> double {
        return garch11::LogLikelihood(params, r2.data(), n_data, cfg, &stats,
                                      garch11::BackcastMethod::TriangularEarly, 0.97);
    };
    // Run Nelder-Mead optimization
    std::vector<double> result = NelderMead(obj_func, N, num_iters, max_iters, tolerance, s);

    // Output results
    std::cout << "GARCH(1,1) Parameters" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Omega = " << result[0] << std::endl;
    std::cout << "Alpha = " << result[1] << std::endl;
    std::cout << "Beta = " << result[2] << std::endl;
    std::cout << "Persistence = " << result[1] + result[2] << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Log Likelihood value = " << result[3] << std::endl;
    std::cout << "Number of iterations = " << result[4] << std::endl;
    std::cout << "Clamping events = " << stats.sigma2_clamps << std::endl;

    // Verify parameter recovery
    std::cout << "--------------------------------" << std::endl;
    std::cout << "True Parameters (for reference)" << std::endl;
    std::cout << "True Omega = " << true_params[0] << std::endl;
    std::cout << "True Alpha = " << true_params[1] << std::endl;
    std::cout << "True Beta = " << true_params[2] << std::endl;
    std::cout << "True Persistence = " << true_params[1] + true_params[2] << std::endl;

    return 0;
}
