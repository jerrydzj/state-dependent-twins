#!/usr/bin/env bash
# Set up a fresh Linux machine with an NVIDIA GPU for twin runs. From the repo root, after cloning the repo and
# copying data/data.csv into data/:
#
#     HF_TOKEN=hf_... bash scripts/gpu_setup.sh
#
# Installs uv, Python 3.14 and the pinned packages into .venv, builds the clean table, downloads Llama 3.1 8B at its
# pinned revision, and runs the tests.
set -euo pipefail

[[ -n "${HF_TOKEN:-}" ]] || { echo 'set HF_TOKEN to a Hugging Face read token'; exit 1; }
[[ -f data/data.csv ]] || { echo 'copy data/data.csv first (see data/README.md)'; exit 1; }

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[twin,dev]'

.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "no CUDA GPU"; print(torch.cuda.get_device_name(0))'
.venv/bin/python -m sdt.mvp.clean
.venv/bin/python -c '
from huggingface_hub import snapshot_download
from sdt.mvp.twin import MODELS
name = "meta-llama/Llama-3.1-8B-Instruct"
print(snapshot_download(name, revision=MODELS[name], allow_patterns=["*.json", "*.safetensors"]))'
.venv/bin/python -m pytest -q
