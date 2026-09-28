"""Where does FeNNiX time go? Time jit(full energy+forces) vs jit(preprocessing only) in JAX (fennix env)."""
import sys, time, numpy as np, jax, jax.numpy as jnp
from fennol import FENNIX
K = int(sys.argv[1])
d = np.load(f'/home/rat/PycharmProjects/ML_models_NAMD/namd_benchmarks/models/fennix/w{K}/reference.npz')
Z, x = d['Z'], d['coordinates']; n = len(Z)
m = FENNIX.load('/home/rat/PycharmProjects/ML_models_NAMD/models/fennix-bio1S.fnx')
raw = dict(species=Z, coordinates=x, natoms=np.array([n], np.int32), batch_index=np.zeros(n, np.int32), total_charge=np.array([0], np.int32))
m.preprocess(**raw); st = m.preproc_state
sp, na, bi, tc = jnp.asarray(Z), jnp.array([n], jnp.int32), jnp.zeros(n, jnp.int32), jnp.array([0], jnp.int32)
def pre(c):
    return m.preprocessing.process(st, dict(species=sp, coordinates=c, natoms=na, batch_index=bi, total_charge=tc))
def full(c):
    e, f, _ = m._energy_and_forces(m.variables, pre(c)); return e, f
def pre_only(c):
    p = pre(c); g = p['graph']
    return g['d12'].sum() + g['edge_src'].sum() + p['graph_filter']['d12'].sum() if 'graph_filter' in p else g['d12'].sum()
def graph_only(c):   # just the all-pairs GraphGenerator part
    lay = m.preprocessing.layers[0]
    out = lay.process(st['layers_state'][0], dict(species=sp, coordinates=c, natoms=na, batch_index=bi, total_charge=tc))
    return out['graph']['d12'].sum() + out['graph']['edge_src'].sum()
print(K, 'keys', list(pre(jnp.asarray(x)).keys()))
xc = jnp.asarray(x)
for name, f in [('full', full), ('pre', pre_only), ('graph', graph_only)]:
    jf = jax.jit(f); jax.block_until_ready(jf(xc))
    ts = []
    for i in range(30):
        t = time.perf_counter(); jax.block_until_ready(jf(xc + 0.001 * i)); ts.append(time.perf_counter() - t)
    print(f'{n:5d} atoms {name:6s} median {1e3*np.median(ts):7.3f} ms  p10 {1e3*np.percentile(ts,10):7.3f}')
