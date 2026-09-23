/**
 ***  C ABI for the NAMD MLFF libtorch shim.
 ***
 ***  Built as a standalone shared library (libnamd_mlff.so) that links
 ***  libtorch in C++ and exposes this flat C interface.  NAMD dlopen's the
 ***  library (RTLD_LOCAL | RTLD_NOW) and calls through this ABI, so the NAMD
 ***  binary itself carries no libtorch link dependency, no torch headers, and
 ***  no torch symbols in its global namespace, exactly the way the FeNNol
 ***  backend treats the PJRT CUDA plugin.
 ***
 ***  Design rules for this boundary:
 ***    - Only POD types cross it: C arrays of double / int64, plain ints,
 ***      C strings, and one opaque handle.  No C++ types, no torch types.
 ***    - No exception may propagate across it.  Every entry point catches
 ***      everything and reports via the return code + mlff_last_error().
 ***    - The shim owns all torch state (the Module, the persistent device and
 ***      pinned-host buffers, the CUDA stream).  Callers pass raw host arrays.
 **/

#ifndef MLFF_SHIM_H
#define MLFF_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-model handle.  One MlffModel owns one loaded TorchScript model,
 * pinned to one device, plus its resident input/output buffers. */
typedef struct MlffModel MlffModel;

/* ---- Capability queries (no handle required) -------------------------- */

/* 1 if libtorch reports a usable CUDA runtime, else 0. */
int mlff_cuda_available(void);

/* Number of CUDA devices libtorch can see (0 on a CPU-only build/host). */
int mlff_device_count(void);

/* ---- Lifecycle -------------------------------------------------------- */

/**
 * Load a TorchScript model and make it resident.
 *
 * @param model_path      Path to the .pt TorchScript file.
 * @param gpu_id          CUDA device ordinal to pin to; < 0 selects CPU.
 * @param supports_batch  Out: set to 1 if the model exposes a forward_batch
 *                        method (so mlff_eval_batch is usable), else 0.
 *                        May be NULL if the caller does not care.
 * @return  Non-NULL handle on success; NULL on failure (see mlff_last_error).
 */
MlffModel *mlff_load(const char *model_path, int gpu_id, int *supports_batch);

/* Release a handle and all the torch state it owns.  NULL is a no-op. */
void mlff_free(MlffModel *m);

/* ---- Evaluation ------------------------------------------------------- */

/**
 * Single-structure forward pass: model.forward(coords, Z, pc_xyz, pc_q).
 *
 * Inputs (host memory, row-major):
 *   coords   [numQM*3] float64  QM atom positions, Angstrom (x,y,z per atom)
 *   Z        [numQM]   int64    atomic numbers
 *   pc_xyz   [numPC*3] float64  point-charge positions (NULL allowed if numPC==0)
 *   pc_q     [numPC]   float64  point-charge values    (NULL allowed if numPC==0)
 * Outputs (caller-allocated host memory):
 *   energy_out  [1]        float64
 *   forces_out  [numQM*3]  float64  (forces = -dE/dx via the model's autograd)
 *   charges_out [numQM]    float64  partial charges; NULL to skip
 *
 * The shim copies the inputs to its device buffers, runs the model with the
 * coordinate leaf marked requires_grad (so the model's internal
 * autograd.grad(energy, coords) yields forces), copies the outputs back, and
 * synchronizes once.  Returns 0 on success, non-zero on failure.
 */
int mlff_eval(MlffModel *m,
              const double *coords, const int64_t *Z, int numQM,
              const double *pc_xyz, const double *pc_q, int numPC,
              double *energy_out, double *forces_out, double *charges_out);

/**
 * Batched forward pass: model.forward_batch(coords, Z, batch, ptr, ...),
 * for B structures concatenated into N_total atoms.
 *
 * Inputs (host memory):
 *   coords  [N_total*3] float64
 *   Z       [N_total]   int64
 *   batch   [N_total]   int64   per-atom structure index in [0, B)
 *   ptr     [B+1]       int64   CSR offsets into the per-atom arrays
 * Outputs (caller-allocated):
 *   energies_out [B]          float64
 *   forces_out   [N_total*3]  float64
 *   charges_out  [N_total]    float64  NULL to skip
 *
 * Requires the model to expose forward_batch (see mlff_load's supports_batch).
 * Returns 0 on success, non-zero on failure.
 */
int mlff_eval_batch(MlffModel *m,
                    int B, int N_total,
                    const double *coords, const int64_t *Z,
                    const int64_t *batch, const int64_t *ptr,
                    double *energies_out, double *forces_out,
                    double *charges_out);

/* ---- Periodic (PBC) evaluation ---------------------------------------- */
/*
 * These are ADDITIONS, not replacements.  mlff_eval / mlff_eval_batch above
 * keep their exact signatures and behaviour, so a NAMD build that predates
 * this header still resolves everything it looks for, and an older
 * libnamd_mlff.so that lacks the symbols below still satisfies a newer NAMD
 * (which resolves them with dlsym and falls back when they are absent).
 *
 * Cell layout, everywhere in this ABI: 9 doubles, row-major basis vectors in
 * Angstrom, a = cell[0..2], b = cell[3..5], c = cell[6..8].  This is the
 * row-vector convention used by ASE, MACE and NequIP, so a fractional
 * coordinate f maps to Cartesian as x = f . M.  A NULL cell means
 * "non-periodic" and must reproduce the mlff_eval result exactly.
 */

/**
 * 1 if the loaded model accepts a cell argument (i.e. was exported against the
 * periodic wrapper contract), else 0.  Check this before passing a non-NULL
 * cell.  Handing a cell to a model that cannot take one is an error rather
 * than something to ignore: dropping it gives you a non-periodic trajectory
 * that looks plausible and is wrong.
 */
int mlff_supports_pbc(MlffModel *m);

/**
 * As mlff_eval, plus a periodic cell.
 *
 * @param cell  9 doubles (see layout above), or NULL for a non-periodic
 *              evaluation.  With NULL this is bit-for-bit mlff_eval.
 * @return 0 on success.  Non-zero if the model has no periodic entry point
 *         (check mlff_supports_pbc first) or the forward failed.
 */
int mlff_eval_pbc(MlffModel *m,
                  const double *coords, const int64_t *Z, int numQM,
                  const double *pc_xyz, const double *pc_q, int numPC,
                  const double *cell,
                  double *energy_out, double *forces_out, double *charges_out);

/**
 * As mlff_eval_batch, plus one cell per structure.
 *
 * @param cells  B*9 doubles, structure b's cell at cells[9*b .. 9*b+8], or NULL
 *               for a non-periodic batch.  Per-structure rather than one shared
 *               cell because concurrent replicas under constant pressure drift
 *               to different box sizes, and collapsing them to one cell would
 *               silently mis-image every replica but one.
 */
int mlff_eval_batch_pbc(MlffModel *m,
                        int B, int N_total,
                        const double *coords, const int64_t *Z,
                        const int64_t *batch, const int64_t *ptr,
                        const double *cells,
                        double *energies_out, double *forces_out,
                        double *charges_out);

/* ---- Periodic virial --------------------------------------------------- */
/*
 * Additions, resolved with dlsym like the _pbc pair above.
 *
 * The virial is the model's strain derivative, 9 doubles row-major, in the
 * model's energy units.  It only exists for a periodic system, hence the cell.
 * `have_virial_out` separates "no virial" from "a virial of zero"; models
 * exported before the virial existed return three outputs and set it to 0.
 */

int mlff_eval_virial(MlffModel *m,
                     const double *coords, const int64_t *Z, int numQM,
                     const double *pc_xyz, const double *pc_q, int numPC,
                     const double *cell,
                     double *energy_out, double *forces_out, double *charges_out,
                     double *virial_out, int *have_virial_out);

/* Batched form.  virials_out is B*9 doubles, structure b's at 9*b. */
int mlff_eval_batch_virial(MlffModel *m,
                           int B, int N_total,
                           const double *coords, const int64_t *Z,
                           const int64_t *batch, const int64_t *ptr,
                           const double *cells,
                           double *energies_out, double *forces_out,
                           double *charges_out,
                           double *virials_out, int *have_virial_out);

/* ---- Diagnostics ------------------------------------------------------ */

/**
 * Human-readable description of the most recent failure on the calling
 * thread.  Never NULL; returns an empty string if no error has occurred.
 * The storage is thread-local and overwritten by the next failing call.
 */
const char *mlff_last_error(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MLFF_SHIM_H */
