/**
 ***  Standalone validation harness for the NAMD MLFF libtorch shim.
 ***
 ***  This program links NO libtorch.  It dlopen's libnamd_mlff.so exactly the
 ***  way NAMD will (RTLD_NOW | RTLD_LOCAL), resolves the C ABI by name, loads a
 ***  TorchScript model, and runs single + batched evaluation.  Its purpose is
 ***  to prove the torch->C boundary works, in particular that forces produced
 ***  by the model's internal autograd.grad(energy, coords) survive crossing
 ***  the boundary (the from_blob device leaf + requires_grad path).
 ***
 ***  Usage:
 ***    mlff_shim_test <libnamd_mlff.so> <model.pt> [gpu_id] [analytic]
 ***
 ***  With the "analytic" flag the model is assumed to be the deterministic
 ***  stub (energy = sum(coords^2), forces = 2*coords) and the results are
 ***  checked against that closed form.  Without it, outputs are only checked
 ***  for finiteness and non-zero forces (enough to confirm grad flowed).
 **/

#include "mlff_shim.h"

#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

/* Function-pointer types mirroring the C ABI (this is the table NAMD's
 * MLFFThread will build in Phase 2). */
using cuda_avail_fn  = int (*)(void);
using dev_count_fn   = int (*)(void);
using load_fn        = MlffModel *(*)(const char *, int, int *);
using free_fn        = void (*)(MlffModel *);
using eval_fn        = int (*)(MlffModel *, const double *, const int64_t *, int,
                               const double *, const double *, int,
                               double *, double *, double *);
using eval_batch_fn  = int (*)(MlffModel *, int, int, const double *,
                               const int64_t *, const int64_t *, const int64_t *,
                               double *, double *, double *);
using last_error_fn  = const char *(*)(void);

template <class T>
bool resolve(void *h, const char *name, T &out) {
    out = reinterpret_cast<T>(dlsym(h, name));
    if (!out) { std::fprintf(stderr, "FAIL: dlsym(%s): %s\n", name, dlerror()); }
    return out != nullptr;
}

bool finite_all(const double *p, int n) {
    for (int i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <libnamd_mlff.so> <model.pt> [gpu_id] [analytic]\n", argv[0]);
        return 64;
    }
    const char *shimPath = argv[1];
    const char *modelPath = argv[2];
    const int   gpuId   = (argc > 3) ? std::atoi(argv[3]) : -1;
    const bool  analytic = (argc > 4) && std::strcmp(argv[4], "analytic") == 0;

    /* The whole point: load exactly as NAMD would. */
    void *h = dlopen(shimPath, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "FAIL: dlopen(%s): %s\n", shimPath, dlerror()); return 1; }

    cuda_avail_fn  cuda_available = nullptr;
    dev_count_fn   device_count   = nullptr;
    load_fn        load           = nullptr;
    free_fn        freeModel      = nullptr;
    eval_fn        eval           = nullptr;
    eval_batch_fn  eval_batch     = nullptr;
    last_error_fn  last_error     = nullptr;
    bool ok = resolve(h, "mlff_cuda_available", cuda_available)
            & resolve(h, "mlff_device_count",   device_count)
            & resolve(h, "mlff_load",           load)
            & resolve(h, "mlff_free",           freeModel)
            & resolve(h, "mlff_eval",           eval)
            & resolve(h, "mlff_eval_batch",     eval_batch)
            & resolve(h, "mlff_last_error",     last_error);
    if (!ok) return 2;

    std::printf("shim loaded: cuda_available=%d device_count=%d\n",
                cuda_available(), device_count());

    int supportsBatch = 0;
    MlffModel *m = load(modelPath, gpuId, &supportsBatch);
    if (!m) { std::fprintf(stderr, "FAIL: mlff_load: %s\n", last_error()); return 3; }
    std::printf("model loaded: '%s' gpu=%d supports_batch=%d\n",
                modelPath, gpuId, supportsBatch);

    int failures = 0;

    /* ---- single structure: a real water geometry (Z = O,H,H) ---- */
    const int numQM = 3;
    const double coords[9] = {
        0.000, 0.000, 0.000,
        0.957, 0.000, 0.000,
       -0.240, 0.927, 0.000,
    };
    const int64_t Z[3] = {8, 1, 1};
    double energy = 0.0, forces[9] = {0}, charges[3] = {0};

    int rc = eval(m, coords, Z, numQM, nullptr, nullptr, 0, &energy, forces, charges);
    if (rc != 0) { std::fprintf(stderr, "FAIL: mlff_eval rc=%d: %s\n", rc, last_error()); ++failures; }
    else {
        std::printf("single: energy=%.6f forces[0]=(%.6f,%.6f,%.6f) charges[0]=%.4f\n",
                    energy, forces[0], forces[1], forces[2], charges[0]);
        if (!std::isfinite(energy) || !finite_all(forces, 9)) {
            std::fprintf(stderr, "FAIL: non-finite single outputs\n"); ++failures;
        }
        double fmag = 0; for (int i = 0; i < 9; ++i) fmag += forces[i] * forces[i];
        if (fmag == 0.0) { std::fprintf(stderr, "FAIL: forces all zero (grad did not flow)\n"); ++failures; }

        if (analytic) {
            double expE = 0; for (int i = 0; i < 9; ++i) expE += coords[i] * coords[i];
            if (std::fabs(energy - expE) > 1e-6) {
                std::fprintf(stderr, "FAIL: energy %.8f != expected %.8f\n", energy, expE); ++failures;
            }
            for (int i = 0; i < 9; ++i) {
                if (std::fabs(forces[i] - 2.0 * coords[i]) > 1e-6) {
                    std::fprintf(stderr, "FAIL: force[%d] %.8f != 2*coord %.8f\n",
                                 i, forces[i], 2.0 * coords[i]); ++failures; break;
                }
            }
            if (failures == 0) std::printf("  analytic single check PASSED\n");
        }
    }

    /* ---- batched: two waters concatenated (N=6, B=2) ---- */
    if (supportsBatch) {
        const int B = 2, N = 6;
        double bcoords[18];
        for (int s = 0; s < 2; ++s)
            std::memcpy(bcoords + s * 9, coords, sizeof(coords));
        /* perturb the second copy so the two structures differ */
        for (int i = 9; i < 18; ++i) bcoords[i] += 0.01 * (i - 9);
        const int64_t bZ[6]   = {8, 1, 1, 8, 1, 1};
        const int64_t bbat[6] = {0, 0, 0, 1, 1, 1};
        const int64_t bptr[3] = {0, 3, 6};
        double energies[2] = {0}, bforces[18] = {0}, bcharges[6] = {0};

        int brc = eval_batch(m, B, N, bcoords, bZ, bbat, bptr, energies, bforces, bcharges);
        if (brc != 0) { std::fprintf(stderr, "FAIL: mlff_eval_batch rc=%d: %s\n", brc, last_error()); ++failures; }
        else {
            std::printf("batch:  energies=(%.6f,%.6f) bforces[0]=%.6f\n",
                        energies[0], energies[1], bforces[0]);
            if (!finite_all(energies, 2) || !finite_all(bforces, 18)) {
                std::fprintf(stderr, "FAIL: non-finite batch outputs\n"); ++failures;
            }
            if (analytic) {
                for (int b = 0; b < B; ++b) {
                    double expE = 0; for (int i = b * 9; i < b * 9 + 9; ++i) expE += bcoords[i] * bcoords[i];
                    if (std::fabs(energies[b] - expE) > 1e-6) {
                        std::fprintf(stderr, "FAIL: batch energy[%d] %.8f != %.8f\n", b, energies[b], expE); ++failures;
                    }
                }
                for (int i = 0; i < 18; ++i) {
                    if (std::fabs(bforces[i] - 2.0 * bcoords[i]) > 1e-6) {
                        std::fprintf(stderr, "FAIL: batch force[%d] %.8f != %.8f\n", i, bforces[i], 2.0 * bcoords[i]); ++failures; break;
                    }
                }
                if (failures == 0) std::printf("  analytic batch check PASSED\n");
            }
        }
    } else {
        std::printf("batch:  skipped (model has no forward_batch)\n");
    }

    freeModel(m);
    dlclose(h);

    std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 10;
}
