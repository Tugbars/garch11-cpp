# GARCH(1,1) Estimation Using Nelder–Mead Optimization

This project demonstrates a **numerically stable GARCH(1,1)** log-likelihood evaluation and parameter estimation using the **Nelder–Mead simplex algorithm** implemented in C++.
It provides an end-to-end workflow — from generating synthetic GARCH data to recovering the model parameters via numerical optimization.

---

## 🧩 Overview

A **GARCH(1,1)** model describes conditional heteroskedasticity in time series data:

[
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
]

where:

* ( \omega ) — long-run variance (constant term)
* ( \alpha ) — short-run volatility response (ARCH effect)
* ( \beta ) — persistence of volatility (GARCH effect)

The goal is to **estimate** ([ \omega, \alpha, \beta ]) by minimizing the **negative log-likelihood (NLL)** of observed squared returns.

---

## ⚙️ Components

### `garch11.hpp`

Implements a **complete and policy-driven GARCH(1,1)** computation suite:

* Numerically stable NLL and Jacobian evaluation
* Configurable clamping (`GarchConfig`, `ClampStats`) to prevent underflow and subnormal arithmetic
* Multiple **backcast initialization** methods for pre-sample variance estimation

  * Mean
  * Triangular (early-weighted)
  * EWMA (exponentially weighted moving average)
* Compile-time policy templates:

  * `InputPolicy::ReturnInf` / `InputPolicy::Throw` for invalid parameters
  * `PredictPolicy::Clamp` / `PredictPolicy::Throw` for forecasting behavior
* Predictive simulation (`predict()`) for future volatility paths

All computations are written in a **header-only, numerically safe, and optimizer-friendly** style with adaptive epsilon scaling and clamping diagnostics.

---

### `neldermead.hpp`

A simple and self-contained **Nelder–Mead simplex optimizer** supporting:

* Reflection, expansion, contraction, and shrink steps
* Configurable parameters (`alpha`, `gamma`, `rho`, `sigma`)
* Termination via tolerance on function spread
* Returns `[x_best, f_best, num_iters]`
* No dependencies beyond the C++ standard library

The optimizer is written generically, accepting any callable `f(std::vector<double>)`.

---

### `main.cpp`

Drives the estimation process:

1. **Generates synthetic GARCH(1,1) data** with known parameters.
2. **Initializes a simplex** around reasonable starting guesses.
3. **Minimizes the negative log-likelihood** via Nelder–Mead.
4. **Displays recovered parameters**, likelihood value, and diagnostic statistics.

---

## Building and Running

### Requirements

* C++17 or later
* Standard C++ toolchain (e.g., GCC, Clang, or MSVC)

### Build

```bash
g++ -O3 -std=c++17 main.cpp -o garch_nm
```

### Run

```bash
./garch_nm
```

---

## 📊 Example Output

```
GARCH(1,1) Parameters
--------------------------------
Omega = 1.9951e-05
Alpha = 0.0987
Beta = 0.8513
Persistence = 0.9500
--------------------------------
Log Likelihood value = 2341.28
Number of iterations = 312
Clamping events = 0
--------------------------------
True Parameters (for reference)
True Omega = 2e-05
True Alpha = 0.1
True Beta = 0.85
True Persistence = 0.95
```

*(Values will vary slightly depending on random seed and tolerance.)*

---

## Key Features and Design Choices

* **Adaptive epsilon scaling** — ensures stable log and division operations even in low-volatility regimes.
* **Compile-time policy templates** — switch between throwing or returning `inf` for invalid input **without runtime overhead**.
* **Modular structure** — `garch11.hpp` can be reused as a standalone GARCH engine in other projects.
* **No external dependencies** — 100% standard C++.
* **Diagnostic counters** — track how often variance clamping occurs (`ClampStats`).

---

## Notes

* The Nelder–Mead optimizer is **derivative-free**, making it suitable for non-smooth likelihoods or noisy objective functions.
* For production-level estimation, you may combine this implementation with more sophisticated optimizers (e.g., BFGS) using the provided analytical gradients in `calc_jac_*`.
* Backcast initialization improves convergence for small sample sizes.

---


