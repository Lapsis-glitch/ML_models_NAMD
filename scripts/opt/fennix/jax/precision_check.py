"""FeNNiX energy/force precision: jit fp32 (as exported) vs jit fp32 'highest' matmul precision vs fp64 reference.
usage: precision_check.py K   (fennix env)"""
import sys, numpy as np, jax, jax.numpy as jnp
K = int(sys.argv[1])
from fennol import FENNIX
d = np.load(f'/home/rat/PycharmProjects/ML_models_NAMD/namd_benchmarks/models/fennix/w{K}/reference.npz')
Z, x = d['Z'], d['coordinates']; n = len(Z)
def build(dtype):
    m = FENNIX.load('/home/rat/PycharmProjects/ML_models_NAMD/models/fennix-bio1S.fnx')
    raw = dict(species=Z, coordinates=x.astype(dtype), natoms=np.array([n], np.int32), batch_index=np.zeros(n, np.int32), total_charge=np.array([0], np.int32))
    m.preprocess(**raw); st = m.preproc_state
    var = jax.tree_util.tree_map(lambda a: a.astype(dtype) if hasattr(a, 'dtype') and jnp.issubdtype(a.dtype, jnp.floating) else a, m.variables)
    sp, na, bi, tc = jnp.asarray(Z), jnp.array([n], jnp.int32), jnp.zeros(n, jnp.int32), jnp.array([0], jnp.int32)
    def f(c):
        pre = m.preprocessing.process(st, dict(species=sp, coordinates=c, natoms=na, batch_index=bi, total_charge=tc))
        e, fo, _ = m._energy_and_forces(var, pre); return e, fo
    return jax.jit(f)
f32 = build(np.float32)
e32, F32 = f32(jnp.asarray(x))
with jax.default_matmul_precision('highest'):
    f32h = build(np.float32); e32h, F32h = f32h(jnp.asarray(x))
jax.config.update('jax_enable_x64', True)
f64 = build(np.float64); e64, F64 = f64(jnp.asarray(x.astype(np.float64)))
kc = 23.0621
for name, e, F in [('fp32 default (shipped)', e32, F32), ('fp32 highest', e32h, F32h)]:
    print(f'{n} atoms {name:24s} dE vs fp64 {kc*abs(float(e[0])-float(e64[0])):9.4f} kcal/mol   max dF {kc*float(jnp.abs(F.astype(jnp.float64)-F64).max()):.4e} kcal/mol/A')
print('E64 kcal', kc*float(e64[0]))
