device cpu
matmul_prec highest
model_file /home/rat/PycharmProjects/ML_models_NAMD/models/fennix-bio1S.fnx
xyz_input{
  file system.xyz
  indexed yes
  has_comment_line no
}
nsteps = 10
dt[fs] = 0.5
traj_format xyz
nblist_skin 2.
tdump[ps] = 1.
nprint = 2
nsummary = 100
thermostat LGV
temperature = 300
gamma[THz] = 10
