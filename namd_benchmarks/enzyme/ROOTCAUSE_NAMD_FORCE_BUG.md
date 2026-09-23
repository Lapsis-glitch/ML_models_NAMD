# Root cause: NAMD applies corrupted QM forces under `qmReplaceAll` (multi-patch)

Companion to `HANDOFF_NAMD_FORCE_BUG.md`. Reproduced, traced, and explained on
2026-06-05. **The bug is NOT in the FeNNiX/MLFF backend or the PJRT shim.** It is a
pre-existing indexing bug in NAMD's shared `qmReplaceAll` force-replacement path,
triggered whenever the replaced (QM) region spans more than one home patch.

## TL;DR

`ComputeQM.C::saveResults` hands the **global, node-wide** `oldForces` array
(indexed by the cumulative `homeIndx` across all patches on the PE) to
`HomePatch::replaceForces()` for **every** patch. But the consumer
(`HomePatch.C:408-414`) indexes that array with the **per-patch-local** atom
index `i = 0..numAtoms-1`. So only the *first* patch reads the right slice; every
later patch reads `oldForces[0..n-1]` again, i.e. the **first patch's forces**.

Result: each atom gets the force belonging to the atom at the same *patch-local
position* in patch 0 (and the other low-index patches). Real force magnitudes,
wrong atoms, with heavy many-to-one reuse.

## The two code sites

Correct contract, as implemented by the other caller `ComputeExt.C:302-317`:
`replacementForces` is reset to `results_ptr` **at the start of each patch** and
`results_ptr` advances one slot per atom, so each patch gets a pointer to *its own
contiguous slice* (local index 0 == that patch's atom 0).

```cpp
// ComputeExt.C  (CORRECT)
ExtForce *results_ptr = msg->force;
for (ap = ap.begin(); ap != ap.end(); ap++) {
    ExtForce *replacementForces = results_ptr;   // per-patch base
    for (int i=0; i<numAtoms; ++i) { ...; ++results_ptr; }
    if (replace) (*ap).p->replaceForces(replacementForces);
}
```

```cpp
// HomePatch.C:408-414  (consumer — indexes by per-patch-local i)
if ( replacementForces ) {
  for ( int i = 0; i < numAtoms; ++i ) {
    if ( replacementForces[i].replace ) {
      for ( int j = 0; j < Results::maxNumForces; ++j ) { f[j][i] = 0; }
      f[Results::normal][i] = replacementForces[i].force;
    }
  }
}
```

```cpp
// ComputeQM.C:2760-2779  (BUG — passes the global base to every patch)
int homeIndxIter = 0;
for (ap = ap.begin(); ap != ap.end(); ap++) {
    ...
    for (int i=0; i<localNumAtoms; ++i) {
        f[i] += oldForces[homeIndxIter].force;   // correct: cumulative index
        ++homeIndxIter;
    }
    if ( callReplaceForces )
        (*ap).p->replaceForces(oldForces);       // BUG: should be oldForces + patchStart
    ...
}
```

Note the accumulation at `f[i] += oldForces[homeIndxIter]` is correct (cumulative
index). But for `replace==1` atoms the consumer **zeroes all force categories and
overwrites** `f[normal][i]` with the wrong-atom value, discarding the correct
accumulation. Under `qmReplaceAll` every atom is `replace==1`, so every atom is
clobbered.

## Why energy is right but forces are wrong

Energy is summed from the model output in `storeQMRes` (`resMsg->energyOrig`) and
never flows through `replaceForces`. It is exactly correct every step. Only the
per-atom force scatter is corrupted, downstream of a correct `backend->evaluate()`.

## Why it looked backend-specific ("MACE/FeNNiX fail, ORCA works")

The bug lives in the **shared** path and triggers only when **`qmReplaceAll` is on
AND the QM region spans >1 patch**. Full-ML runs replace the *whole solvated
system* (thousands of atoms across many patches) -> triggers hard. Classic
ORCA/MOPAC QM/MM uses a *small* QM region that lands in a single patch (patch 0,
offset 0) -> the bug is invisible. It is not an in-process-vs-file-based
distinction; ORCA would corrupt identically with a large multi-patch
`qmReplaceAll` region.

## Evidence (all reproducible, no NAMD rebuild)

System: `systems/mono_shell5` (4525 atoms, full-ML, `qmReplaceAll on`,
`QMSoftware fennol`). T=0 NVE probe -> `pos.dcd` + `force.dcd`.

1. **Symptom reproduced** (`diagnostics/lag_proof.py`):
   `corr(force.dcd[step1], F_model(x0)) = 0.019`; `corr(force.dcd, m*dx) = 0.9998`
   (the DCD really is the applied force); energy matches model to 4 digits.

2. **Not multi-PE**: `+p1` reproduces identically (corr 0.019). The system still
   has multiple patches on the one PE.

3. **Not the PJRT read / device layout**: the standalone C++ probe on the *same*
   N=4525 artifact reads `outputs[1]` correctly -- forces max abs diff
   6.24e-3 eV/A vs the JAX reference (tol 2.9e-2). Same `copy_buffer_to_host`
   code as NAMD. Not `memory_fraction` either (FENNIX_MEM_FRACTION=0.7 -> no change).

4. **Not GPU-resident**: CUDASOAintegrate is off (CPU integration).

5. **Direct fingerprint of the tiling collapse** (`force.dcd`, step 1):
   - 4525 atoms -> only **409 distinct force vectors** (correct behavior would give
     ~4525 distinct, since model forces are continuous floats). 4486 atoms share
     their vector with another atom.
   - Internal coherence: **max multiplicity = 18 = number of home patches**;
     **#distinct vectors = 409 = largest patch size** (only `oldForces[0..408]`
     are ever read); the per-multiplicity counts are the gaps between the sorted
     patch sizes (all non-negative, self-consistent). This is the signature of
     "tile the low-index slice across every patch by local index".
     (Note: Sum_k k*count_k = 4525 is automatic for any partition and proves
     nothing on its own; the diagnostic facts are the 409 and the coherence above.)

6. **Measured (not inferred): the backend forces are correct, NAMD mis-scatters**
   (set-membership of the 409 distinct applied vectors vs `F_model(x0)`):
   - each of the 409 distinct applied vectors matches a **distinct** real model-
     force atom (408 unique atoms), within coordinate-rounding tolerance
     (mean 0.24, 220/409 within 0.2 kcal/mol/A; the spread is qm.pdb 3-decimal
     coords vs the runtime float32 coords on a stiff force field).
   - **4117 of 4525** atoms' model forces are **never applied to anyone** -- the
     high-`homeIndx` slice that no patch is large enough to reach.
   This directly shows `evaluate()` returned real model forces in-process; the
   corruption is pure mis-assignment, so the fix restores correctness rather than
   merely de-duplicating.

## Fix (one site, restores the ComputeExt contract)

In `ComputeQM.C::saveResults`, give each patch the base pointer to its own slice:

```cpp
int homeIndxIter = 0;
for (ap = ap.begin(); ap != ap.end(); ap++) {
    Results *r = (*ap).forceBox->open();
    Force *f = r->f[Results::normal];
    int localNumAtoms = (*ap).p->getNumAtoms();
    int patchStart = homeIndxIter;                    // ADD
    for (int i=0; i<localNumAtoms; ++i) {
        f[i] += oldForces[homeIndxIter].force;
        ++homeIndxIter;
    }
    if ( callReplaceForces )
        (*ap).p->replaceForces(oldForces + patchStart);   // was: oldForces
    (*ap).forceBox->close(&r);
}
```

`oldForces` is indexed by `homeIndx`, which `doWork` assigns as the cumulative
per-atom counter in `patchList` order; `saveResults` iterates `patchList` in the
same order, so patch k's atoms occupy the contiguous block
`oldForces[patchStart_k .. patchStart_k + n_k)`. The offset makes the per-patch-
local index in the consumer line up with the correct atoms.

## Success criterion (handoff S9): VERIFIED 2026-06-05

Fix applied to `ComputeQM.C::saveResults` and `namd3` relinked (targeted rebuild
of ComputeQM.o + link). Re-ran the T=0 NVE probe (`+p1`) on mono_shell5:

| check | buggy | fixed |
|---|---|---|
| corr(applied force, F_model) per step | 0.019 | **1.0000** (every step) |
| rms\|applied - model\| | 13.4 | **0.023** kcal/mol/A (precision floor) |
| distinct force vectors | 409 / 4525 | **4525 / 4525** |
| NVE total energy | runaway (explodes ~11 fs) | **conserved**: -25835.55 -> -25838.03 over 6 steps (~0.01%) |
| PE from rest (T=0) | rose (impossible) | falls -25835 -> -26144, KE 0 -> 306 (correct downhill) |

The many-to-one tiling collapse is gone (all 4525 forces distinct), applied
forces equal the model's true forces to the coordinate-precision floor, and the
quench conserves total energy instead of exploding. Verification artifacts in
`diagnostics/fixrun/` (probe_fix.log, force.dcd, pos.dcd).
