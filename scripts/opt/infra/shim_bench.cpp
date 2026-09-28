/*
 * shim_bench: time libnamd_mlff.so the way NAMD calls it (dlopen RTLD_LOCAL,
 * C ABI, host double buffers in/out), with MD-like per-call coordinate jitter.
 * Links only -ldl.  All shim env knobs (NAMD_MLFF_*) apply.
 *
 *   shim_bench <libnamd_mlff.so> <model.pt> <geom.pdb|.xyz> [walkers=1] [warmup=25] [iters=200] [jitter=0.02] [gpu=0]
 *
 * walkers>1 uses mlff_eval_batch with W jittered replicas.  Prints one line:
 *   RESULT load_ms=.. warmup_ms=.. median_ms=.. p10_ms=.. p90_ms=.. E0=..
 */
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct MlffModel;
using load_fn  = MlffModel *(*)(const char *, int, int *);
using eval_fn  = int (*)(MlffModel *, const double *, const int64_t *, int,
                         const double *, const double *, int, double *, double *, double *);
using evalb_fn = int (*)(MlffModel *, int, int, const double *, const int64_t *,
                         const int64_t *, const int64_t *, double *, double *, double *);
using err_fn   = const char *(*)(void);

static int sym2z(std::string s) {
    if (s.size() > 1) s = std::string(1, s[0]) + (char)std::tolower(s[1]);
    const char *t[] = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl"};
    for (int i = 0; i < 17; i++) if (s == t[i]) return i + 1;
    return 0;
}

static void readGeom(const std::string &p, std::vector<double> &xyz, std::vector<int64_t> &Z) {
    std::ifstream f(p);
    std::string line;
    if (p.size() > 4 && p.substr(p.size() - 4) == ".xyz") {
        std::getline(f, line); int n = std::stoi(line); std::getline(f, line);
        for (int i = 0; i < n && std::getline(f, line); i++) {
            std::istringstream ss(line); std::string s; double x, y, z; ss >> s >> x >> y >> z;
            Z.push_back(std::isdigit(s[0]) ? std::stoi(s) : sym2z(s));
            xyz.insert(xyz.end(), {x, y, z});
        }
        return;
    }
    while (std::getline(f, line)) {
        if (line.rfind("ATOM", 0) != 0 && line.rfind("HETATM", 0) != 0) continue;
        double x = std::stod(line.substr(30, 8)), y = std::stod(line.substr(38, 8)), z = std::stod(line.substr(46, 8));
        std::string el = line.size() >= 78 ? line.substr(76, 2) : "";
        el.erase(std::remove(el.begin(), el.end(), ' '), el.end());
        if (el.empty()) { el = line.substr(12, 4); el.erase(std::remove_if(el.begin(), el.end(), [](char c){ return !std::isalpha(c); }), el.end()); el = el.substr(0, 1); }
        Z.push_back(sym2z(el));
        xyz.insert(xyz.end(), {x, y, z});
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s shim.so model.pt geom [W] [warmup] [iters] [jitter] [gpu]\n", argv[0]); return 2; }
    int W = argc > 4 ? std::atoi(argv[4]) : 1;
    int warm = argc > 5 ? std::atoi(argv[5]) : 25;
    int iters = argc > 6 ? std::atoi(argv[6]) : 200;
    double jit = argc > 7 ? std::atof(argv[7]) : 0.02;
    int gpu = argc > 8 ? std::atoi(argv[8]) : 0;

    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto load = (load_fn)dlsym(h, "mlff_load");
    auto eval = (eval_fn)dlsym(h, "mlff_eval");
    auto evalb = (evalb_fn)dlsym(h, "mlff_eval_batch");
    auto lerr = (err_fn)dlsym(h, "mlff_last_error");

    std::vector<double> xyz; std::vector<int64_t> Z1;
    readGeom(argv[3], xyz, Z1);
    const int n = (int)Z1.size(), N = n * W;

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    int sb = 0;
    auto t0 = clk::now();
    MlffModel *m = load(argv[2], gpu, &sb);
    auto t1 = clk::now();
    if (!m) { std::fprintf(stderr, "load: %s\n", lerr()); return 1; }

    const int POOL = 64;
    std::mt19937_64 rng(1234);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<double>> pool(POOL, std::vector<double>(N * 3));
    for (auto &c : pool)
        for (int w = 0; w < W; w++)
            for (int i = 0; i < n * 3; i++) c[w * n * 3 + i] = xyz[i] + jit * nd(rng);
    std::vector<int64_t> Z(N), batch(N), ptr(W + 1);
    for (int w = 0; w < W; w++) { for (int i = 0; i < n; i++) { Z[w * n + i] = Z1[i]; batch[w * n + i] = w; } ptr[w] = (int64_t)w * n; }
    ptr[W] = N;
    std::vector<double> e(W), f(N * 3), q(N);

    auto call = [&](int k) {
        const double *c = pool[k % POOL].data();
        int rc = (W == 1) ? eval(m, c, Z.data(), n, nullptr, nullptr, 0, e.data(), f.data(), q.data())
                          : evalb(m, W, N, c, Z.data(), batch.data(), ptr.data(), e.data(), f.data(), q.data());
        if (rc) { std::fprintf(stderr, "eval: %s\n", lerr()); std::exit(1); }
    };
    auto t2 = clk::now();
    for (int k = 0; k < warm; k++) call(k);
    auto t3 = clk::now();
    std::vector<double> ts;
    for (int k = 0; k < iters; k++) { auto a = clk::now(); call(warm + k); ts.push_back(ms(a, clk::now())); }
    call(0);
    double E0 = e[0];
    std::sort(ts.begin(), ts.end());
    std::printf("RESULT load_ms=%.1f warmup_ms=%.1f median_ms=%.3f p10_ms=%.3f p90_ms=%.3f E0=%.6f n=%d W=%d\n",
                ms(t0, t1), ms(t2, t3), ts[ts.size() / 2], ts[ts.size() / 10], ts[ts.size() * 9 / 10], E0, n, W);
    std::fflush(stdout);
    std::_Exit(0);   /* skip teardown, like a finished NAMD run */
}
