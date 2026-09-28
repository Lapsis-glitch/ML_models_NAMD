device cpu
matmul_prec highest
model_file /home/rat/PycharmProjects/ML_models_NAMD/models/fennix-bio1S.fnx

# Tinker-indexed xyz (index element x y z); non-periodic (no cell) finite droplet,
# matching the NAMD finite-cluster setup we benchmarked.
xyz_input{
  file system.xyz
  indexed yes
  has_comment_line no
}

# neutral (q=0); total_charge defaults to 0

# --- dynamics (FeNNol's recommended protein settings, cf. examples/md/dhfr) ---
nsteps = 40
dt[fs] = 0.5
traj_format xyz
nblist_skin 2.
tdump[ps] = 1.
nprint = 2
nsummary = 50

thermostat LGV
temperature = 300.
gamma[THz] = 10.
