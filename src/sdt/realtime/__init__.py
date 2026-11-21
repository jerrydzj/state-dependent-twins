"""The realtime-benchmark study (docs/prereg/realtime-benchmark.md): LLM twins against simple models, each class
refitted every month and picked by its recent forecasts. Its runs are in experiments/realtime/."""
from sdt import paths

RUNS = paths.EXPERIMENTS / 'realtime'       # one folder per run of this study
