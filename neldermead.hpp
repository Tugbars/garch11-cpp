#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>

struct NMResult {
    std::vector<double> xbest; // best parameters
    double fbest;              // best function value
    int iters;                 // iterations used
};

// Returns: [x0..x_{N-1}, fbest, iters] to match your example indexing
inline std::vector<double> NelderMead(
    const std::function<double(const std::vector<double>&)>& f,
    int N,
    double /*NumIters_unused*/, double MaxIters,
    double Tolerance,
    std::vector<std::vector<double>> s,    // simplex size: N x (N+1) in your example layout
    double alpha = 1.0, double gamma = 2.0, double rho = 0.5, double sigma = 0.5)
{
    // Your input 's' is arranged as s[dim][vertex], convert to vertices list
    std::vector<std::vector<double>> x(N+1, std::vector<double>(N));
    for (int j = 0; j < N+1; ++j)
        for (int i = 0; i < N; ++i)
            x[j][i] = s[i][j];

    std::vector<double> fx(N+1);
    for (int j = 0; j < N+1; ++j) fx[j] = f(x[j]);

    auto sort_simplex = [&](){
        std::vector<int> idx(N+1);
        for (int i = 0; i < N+1; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return fx[a] < fx[b]; });
        std::vector<std::vector<double>> x2(N+1, std::vector<double>(N));
        std::vector<double> fx2(N+1);
        for (int k = 0; k < N+1; ++k) { x2[k] = x[idx[k]]; fx2[k] = fx[idx[k]]; }
        x.swap(x2); fx.swap(fx2);
    };

    auto centroid_except_worst = [&](std::vector<double>& c){
        c.assign(N, 0.0);
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) c[i] += x[j][i];
        for (int i = 0; i < N; ++i) c[i] /= N;
    };

    auto new_point = [&](const std::vector<double>& base,
                         const std::vector<double>& dir, double scale){
        std::vector<double> p(N);
        for (int i = 0; i < N; ++i) p[i] = base[i] + scale * (dir[i] - base[i]);
        return p;
    };

    int iter = 0;
    for (; iter < static_cast<int>(MaxIters); ++iter) {
        sort_simplex();
        double fbest = fx[0], fworst = fx[N];
        double fspread = std::fabs(fworst - fbest);
        if (fspread <= Tolerance) break;

        // Centroid of best..second-worst
        std::vector<double> c; centroid_except_worst(c);

        // Reflection
        std::vector<double> xr(N);
        for (int i = 0; i < N; ++i) xr[i] = c[i] + alpha * (c[i] - x[N][i]);
        double fr = f(xr);

        if (fr < fx[0]) {
            // Expansion
            std::vector<double> xe(N);
            for (int i = 0; i < N; ++i) xe[i] = c[i] + gamma * (xr[i] - c[i]);
            double fe = f(xe);
            if (fe < fr) { x[N] = xe; fx[N] = fe; }
            else         { x[N] = xr; fx[N] = fr; }
        } else if (fr < fx[N-1]) {
            // Accept reflection
            x[N] = xr; fx[N] = fr;
        } else {
            // Contraction
            bool outside = (fr < fx[N]);
            std::vector<double> xc(N);
            if (outside) {
                for (int i = 0; i < N; ++i) xc[i] = c[i] + rho * (xr[i] - c[i]);
            } else {
                for (int i = 0; i < N; ++i) xc[i] = c[i] + rho * (x[N][i] - c[i]);
            }
            double fc = f(xc);
            if (fc < (outside ? fr : fx[N])) {
                x[N] = xc; fx[N] = fc;
            } else {
                // Shrink towards best
                for (int j = 1; j < N+1; ++j) {
                    for (int i = 0; i < N; ++i)
                        x[j][i] = x[0][i] + sigma * (x[j][i] - x[0][i]);
                    fx[j] = f(x[j]);
                }
            }
        }
    }

    // Final order
    std::vector<int> idx(N+1);
    for (int i=0;i<N+1;++i) idx[i]=i;
    std::sort(idx.begin(), idx.end(), [&](int a,int b){return fx[a] < fx[b];});
    std::vector<double> out(N+2);
    for (int i=0;i<N;++i) out[i] = x[idx[0]][i];
    out[N]   = fx[idx[0]];
    out[N+1] = static_cast<double>(iter);
    return out;
}