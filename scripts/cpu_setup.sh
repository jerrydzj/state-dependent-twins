#!/usr/bin/env bash
# Set up a fresh Linux machine for the realtime-benchmark run, which needs no GPU and no PyTorch. From the repo root,
# after cloning the repo and copying data/data.csv into data/:
#
#     bash scripts/cpu_setup.sh
#
# Installs uv, Python 3.14 and the pinned packages without the `twin` extra into .venv, builds the clean table, and
# runs the tests; the tests that need PyTorch skip.
set -euo pipefail

[[ -f data/data.csv ]] || { echo 'copy data/data.csv first (see data/README.md)'; exit 1; }

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[dev]'

.venv/bin/python -m sdt.data.clean
# Cap the numerical libraries' threads at this machine's CPUs: a container often shows every core of its host, and
# threads far beyond the CPUs it may use make the tests run many times slower
CPUS=$(.venv/bin/python -c 'from sdt.experiment import available_cpus; print(available_cpus())')
echo "$CPUS CPUs available"
OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS .venv/bin/python -m pytest -q
