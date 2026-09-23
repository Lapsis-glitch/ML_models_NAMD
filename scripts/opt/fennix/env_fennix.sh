# source me: runtime env for the FeNNiX PJRT benches (same libs NAMD's fennol runs use)
export FENNIX_ENV=${FENNIX_ENV:-/home/rat/miniconda3/envs/fennix}
export NAMD_FENNIX_PLUGIN=${NAMD_FENNIX_PLUGIN:-$FENNIX_ENV/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so}
export LD_LIBRARY_PATH=$(ls -d $FENNIX_ENV/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:):${LD_LIBRARY_PATH:-}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
REPO=/home/rat/PycharmProjects/ML_models_NAMD
BENCH=$REPO/scripts/opt/fennix/bench/bench_pjrt
LOCK=$REPO/scripts/opt/.gpu_bench.lock
