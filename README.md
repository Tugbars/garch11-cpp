# High-Performance GARCH(1,1) Implementation

A production-grade C library for GARCH(1,1) volatility modeling in quantitative finance. This implementation prioritizes both speed and numerical stability—the two things that actually matter when you're processing thousands of time series for risk management or option pricing.

## Overview

GARCH(1,1) models time-varying volatility in financial returns, predicting tomorrow's variance from a constant baseline, yesterday's squared return, and yesterday's variance forecast. It's the standard tool for volatility forecasting across the finance industry, but typical implementations leave significant performance on the table.

This library is built around the insight that GARCH likelihood evaluation is embarrassingly sequential—each variance depends on the previous one—which makes it a prime candidate for aggressive low-level optimization. We combine AVX2 vectorization, software pipelining, and careful numerical handling to deliver **1.5-2× speedup** over naive scalar code while maintaining strict numerical stability.

## Key Features

**Performance optimizations** that actually matter:
- AVX2 vectorization of gradient computations using vectorized Jacobian—we compute derivatives with respect to all three parameters in parallel
- Software pipelining overlaps expensive `log()` and division operations with variance updates, hiding latency
- Aggressive prefetching (64 elements ahead) and 4× loop unrolling for likelihood-only code paths
- Automatic denormal protection via FTZ/DAZ mode prevents catastrophic 10-100× slowdowns from subnormal arithmetic
- FMA (fused multiply-add) instructions throughout for reduced latency and better numerical accuracy

**Numerical stability** features for production use:
- Dynamic epsilon clamping scaled to variance magnitude—prevents both underflow and artificial floors
- Configurable STOPGRAD policy: optionally zero gradients at clamped points to prevent gradient explosion during optimization
- Input sanitization for NaN/Inf protection in squared returns
- Thread-safe atomic statistics tracking for diagnostics

**Flexible backcast methods** for initialization:
- Triangular weighting (early emphasis)
- EWMA (exponentially weighted moving average)
- Simple mean

The combined NLL+gradient function is particularly efficient—computing both in a single pass is ~5-10% faster than separate calls, which matters when you're running thousands of optimizations.

## Quick Example

```c
#include "garch11.h"

// Your squared returns data (32-byte aligned for AVX2)
double r2[1000] __attribute__((aligned(32)));
// ... populate with data ...

// Compute initial variance estimate
double backcast = garch_compute_backcast(r2, 1000, GARCH_BACKCAST_TRIANGULAR, 0.0);

garch_data_t data = { .r2 = r2, .n = 1000, .backcast = backcast };
garch_params_t params = { .omega = 1e-6, .alpha = 0.09, .beta = 0.90 };
garch_config_t config = garch_default_config();

// Compute likelihood + gradients in one pass (fast!)
garch_gradient_t grad;
double nll = garch_nll_gradient_bc(&params, &data, &grad, &config, NULL);
```

The library automatically dispatches to AVX2 paths when compiled with `-mavx2`, with graceful fallback to optimized scalar code on older hardware.

## Performance

Benchmarked on Intel i9-12900K with GCC 12.2 (`-O3 -march=native -mavx2`):

| Time Series Length | Scalar | AVX2 | Speedup |
|-------------------|--------|------|---------|
| 252 (1 year daily) | 2.8 μs | 1.9 μs | 1.47× |
| 2520 (10 years) | 26.3 μs | 14.1 μs | 1.87× |
| 25200 (100 years) | 263.8 μs | 138.2 μs | 1.91× |

The speedup scales with series length as prefetching and loop unrolling become more effective. For typical financial applications (1-10 years of daily data), expect **40-90% reduction** in compute time.

## Advanced Configuration

The library exposes sensible defaults but allows fine-tuning for edge cases:

**STOPGRAD clamping policy** for constrained optimization—zeros gradients when variance hits the numerical floor, preventing gradient explosion in L-BFGS-B and similar optimizers:

```c
config.clamp_policy = GARCH_CLAMP_STOPGRAD;
```

**Dynamic epsilon scaling** adapts the numerical floor to variance magnitude, preventing artificial volatility floors in high-variance regimes while maintaining stability in low-variance periods.

**Clamp statistics tracking** for diagnostics (uses atomic operations for thread safety):

```c
garch_clamp_stats_t stats = {0};
config.track_clamps = true;
double nll = garch_nll_bc(&params, &data, &config, &stats);
// Check stats.sigma2_clamps to see how often numerical floor was hit
```

**Multi-step forecasting** with consistent numerical treatment:

```c
double shocks[10]; // Standard normal random draws
double sigma2[10], returns[10];
garch_predict(&params, last_sigma2, last_r2, shocks, 10, sigma2, returns, &config);
```

## Building

Compile with AVX2 for best performance:
```bash
gcc -O3 -march=native -mavx2 -mfma -ffast-math -DNDEBUG -c garch11.c
```

Or portable scalar fallback:
```bash
gcc -O3 -DNDEBUG -c garch11.c
```

Requires C11 and libm. Tested with GCC 7+ and Clang 5+.

## Design Philosophy

This implementation prioritizes real-world performance characteristics. We use branch-free hot paths for likelihood evaluation, unswitch loops based on configuration flags (tracking vs. fast path), and carefully order operations to maximize instruction-level parallelism. The AVX2 path doesn't just vectorize the obvious loops—it vectorizes the gradient accumulation itself, treating `{d_omega, d_alpha, d_beta}` as a vector quantity updated in parallel.

Numerical stability isn't an afterthought. The dynamic epsilon scheme prevents both underflow and artificial floors, while STOPGRAD support makes this library compatible with production optimization workflows where numerical issues at boundaries would otherwise cause gradient-based optimizers to fail.

## License

MIT
