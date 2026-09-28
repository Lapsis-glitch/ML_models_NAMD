# Diagnostic: `qmReplaceAll off` vs `on` — mono_shell5 (4525 atoms, FeNNiX)

Same conf, same dt=0.25 fs, same 60 steps, only flag flipped. Confirms
`ROOTCAUSE_NAMD_FORCE_BUG.md`: the blow-up is the buggy `replaceForces(oldForces)`
scatter in `ComputeQM.C::saveResults`, which fires ONLY under `qmReplaceAll on`.

| step | `qmReplaceAll on` (buggy path)        | `qmReplaceAll off` (bypasses replaceForces) |
|------|--------------------------------------|---------------------------------------------|
| 0    | TOTAL -22592, TEMP 305               | TOTAL -22655, TEMP 300                       |
| 20   | TOTAL -15101, **TEMP 738**           | TOTAL -22629, TEMP 231                       |
| 40   | TOTAL +35606, **TEMP 3464**          | TOTAL -22587, TEMP 262                       |
| 45   | **RATTLE constraint failure → FATAL**| TOTAL -22578, TEMP 280                       |
| 60   | (already crashed)                    | TOTAL -22570, TEMP 286 → **End of program**  |

- `on`:  TEMP runs away 305→3464 K, total energy +58000 kcal/mol in 40 steps, RATTLE
         fails at step ~45 → crash (rc1, 36 s). This is the corrupted-force signature
         (applied force corr ≈0.02 with the model force; 4525 atoms → 409 distinct vectors).
- `off`: TEMP stays bounded 222–300 K, total energy drifts only ~85 kcal/mol (~0.4 %)
         over 60 steps, QMENERGY oscillates smoothly in [-25837, -25604]. Runs to
         completion cleanly (rc0, WallClock 57.8 s, "End of program").

## Why this isolates the bug
With `qmReplaceAll off` the buggy `replaceForces(oldForces)` call is NOT taken; the QM
forces are applied through the *correct* cumulative index `f[i] += oldForces[homeIndxIter]`.
So the same backend forces, applied at the right atoms, give stable dynamics.

Mechanism confirmed in source (`ComputeQM.C`): `callReplaceForces` is set true only when a
force has `replace==1` (l.2735), and `replace` is set per-atom by `qmReplaceAll` (l.892).
`off` → no atom `replace==1` → `callReplaceForces` stays false → the buggy
`replaceForces(oldForces)` at l.2781 is never reached.

## Proof it's the QM forces (not residual MM) carrying the stable run
**QMENERGY (model PE) and TEMP are anti-correlated** — energy conservation *under the model
potential*: model PE is least-negative at step 26 (−24968, +870 vs start) exactly where TEMP
bottoms (~222 K at ts 20–25); both recover together by ts 50–55. KE↔model-PE exchange is the
signature of a conservative force field correctly applied, which rules out "MM forces masking
a still-broken QM scatter."

## Caveat — `off` is a DIAGNOSTIC, not a production mode
With the whole droplet flagged QM (beta=1), `off` makes NAMD *add* QM forces to whatever
intra-QM MM terms it doesn't exclude, rather than purely replacing — physics is not
guaranteed identical to native `fennol_md`. The real fix is the one-line `oldForces +
patchStart` patch in `ComputeQM.C::saveResults` (see ROOTCAUSE doc), which makes the
correct full-ML `qmReplaceAll on` path stable too. Reproduce:
`diagnostics/qmreplace_off/enzyme_off.conf` + the fennol env from `../env.sh`.
