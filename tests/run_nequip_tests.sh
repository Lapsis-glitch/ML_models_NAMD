#!/usr/bin/env bash
# Run NequIP/Allegro pipeline tests in the nequip_env conda environment.
# Usage: bash tests/run_nequip_tests.sh

set -e

# Ensure conda shell functions are available
eval "$(conda shell.bash hook)"

conda deactivate 2>/dev/null || true
conda activate nequip_env

echo "=== Environment: $CONDA_DEFAULT_ENV ==="
echo "Python: $(which python)"
python -c "import e3nn; print('e3nn:', e3nn.__version__)"
python -c "import nequip; print('nequip:', nequip.__version__)"
echo "========================================="

cd "$(dirname "$0")/.."

python -u -m pytest \
    tests/test_pipeline_integration.py::TestNequIPPipeline \
    tests/test_pipeline_integration.py::TestAllegroPipeline \
    -v --tb=long

