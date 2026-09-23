#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runtime environment for the NAMD3 ML-FF / FeNNol / xTB QM benchmark suite.
# Source before launching anything:  source env.sh
#
# The `namd_fennix` NAMD3 build natively links libtorch (TorchScript MLFF
# backend) and the native xTB library, and dlopens the JAX CUDA PJRT plugin
# for the FeNNol backend.  All three sets of libs must be reachable here or
# namd3 will not even load (the FFTW2 libs in particular are a hard NEEDED
# dependency of the binary).
# ---------------------------------------------------------------------------

# External tree locations. Defaults are derived from $HOME so this SAME file
# works on any machine that mirrors the layout under its home (/home/rat here,
# /home/radu on the remote) with no edits. Override any inline if your layout
# differs, e.g.  COMPILE_ROOT=/opt FENNIX_ENV=/data/envs/fennix ./run_benchmark.sh
COMPILE_ROOT="${COMPILE_ROOT:-$HOME/compile_NAMD_MACE}"
NAMD_ROOT="${NAMD_ROOT:-$COMPILE_ROOT/namd_fennix}"
LIBTORCH_ROOT="${LIBTORCH_ROOT:-$COMPILE_ROOT/libtorch}"
FENNIX_ENV="${FENNIX_ENV:-$HOME/miniconda3/envs/fennix}"
export NAMD_BIN_DIR="$NAMD_ROOT/Linux-x86_64-g++"
export NAMD3="$NAMD_BIN_DIR/namd3"
export CHARMRUN="$NAMD_BIN_DIR/charmrun"

# libtorch (CPU + CUDA) — TorchScript MLFF backend (MACE / ANI / NequIP / SchNet)
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib:${LD_LIBRARY_PATH:-}"
# FFTW2 single-precision libs — hard NEEDED dependency of namd3
export LD_LIBRARY_PATH="$NAMD_ROOT/fftw/lib:${LD_LIBRARY_PATH}"
# native xTB shared library — qmSoftware xtb (GFN2-xTB; dlopen'd via NAMD_XTB_LIB)
export LD_LIBRARY_PATH="$NAMD_ROOT/xtb/build:${LD_LIBRARY_PATH}"
export NAMD_XTB_LIB="$NAMD_ROOT/xtb/build/libxtb.so"

export PATH="$NAMD_BIN_DIR:${PATH}"

# native xTB parameter directory (gfn0/1/2, gfnff)
export XTBPATH="$NAMD_ROOT/xtb:${XTBPATH:-}"

# MLFF libtorch shim — the ONLY thing that links libtorch; namd3 dlopen's it.
# Built once from $NAMD_ROOT/src/mlff_shim/build_and_test.sh into ./lib.
export NAMD_MLFF_LIB="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/lib/libnamd_mlff.so"

# FeNNol PJRT CUDA plugin — qmSoftware fennol (or pass qmFennixPluginPath in conf).
# Globbed under $FENNIX_ENV so it tracks the env's python version (3.11 here).
export NAMD_FENNIX_PLUGIN="${NAMD_FENNIX_PLUGIN:-$(ls -d "$FENNIX_ENV"/lib/python*/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so 2>/dev/null | xargs -r readlink -f | sort -u | head -1)}"

# The JAX PJRT plugin needs ITS OWN bundled CUDA libs (esp. cuDNN 9.14).  These
# must be PREPENDED ahead of libtorch/lib for fennol runs only — prepending them
# for TorchScript (mlff) runs can break libtorch's own CUDA inference.  The
# driver applies $FENNIX_CUDA_LIBS to LD_LIBRARY_PATH only when QMSoftware=fennol.
_FENV_NV="$(ls -d "$FENNIX_ENV"/lib/python*/site-packages/nvidia 2>/dev/null | xargs -r readlink -f | sort -u | head -1)"
export FENNIX_CUDA_LIBS="$(ls -d "$_FENV_NV"/*/lib 2>/dev/null | paste -sd: -)"

echo "[env] namd3      = $NAMD3"
echo "[env] charmrun   = $CHARMRUN"
echo "[env] fennix PJRT= $NAMD_FENNIX_PLUGIN"
