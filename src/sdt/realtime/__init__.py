"""The realtime-benchmark study (docs/prereg/realtime-benchmark.md): LLM twins against simple models. Every candidate
is refitted every month, and each class picks the candidate with the best recent forecasts. Its runs are in
experiments/realtime/."""
from sdt import paths

RUNS = paths.EXPERIMENTS / 'realtime'       # one folder per run of this study
