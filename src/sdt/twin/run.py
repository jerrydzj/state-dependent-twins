"""Run the twins and save their raw option probabilities to a new folder experiments/<study>/<date>-<description>/.

    python -m sdt.twin.run realtime twin-web --months 202404-202407,202408-202609
    python -m sdt.twin.run mvp twin-pilot --months 202404-202407 --stages base,counterfactual,rerun,bf16 --batch-size 64
    python -m sdt.twin.run mvp speed-test --benchmark          # prompts per second and GPU memory per batch size
    python -m sdt.twin.run realtime <folder name> --resume ...  # continue an interrupted run at the same commit

*Running* a prompt means the model computes its option probabilities; *scoring* compares probabilities with answers.
Every interview of a block is run for every item, whether or not it answered the item, so no run depends on an answer;
interviews without an answer are left out only when scoring.

Blocks: `--months` lists blocks of survey months, each written FIRST-LAST. A block holds the web interviews of its
months, in the clean table's order, and is named by its months. Prompts are batched within a block, so two runs with
the same blocks put every prompt in the same batch.

Stages, in this order:

1. `base`: every interview of each block, for each item and feature set, at --batch-size.
2. `counterfactual`: the first block's interviews, for each item, on demographics_macro_politics, under each scenario
   of SCENARIOS.
3. `rerun`: 10 prompts of the first block per item and feature set, drawn at random (seed 0), each run alone (batch
   size 1). Their gap from the base stage measures what batching changes.
4. `bf16`: the same prompts, each run alone with the weights kept in bf16. Their gap from `rerun` measures what fp32
   changes.

The raw option probabilities are written chunk by chunk (stage x scenario x block x feature set x item) to chunks/, so
an interrupted run can continue where it stopped. When every chunk is done, the run writes:

- option_probs.parquet: the raw probabilities `p1` to `p3` of every prompt run, with its stage, scenario, block (column
  `interviews`), feature set, item and CASEID (not in git: a derivative of the microdata).
- run.json: commit, data hash, model and revision, device, package versions, settings, prompts and seconds per stage,
  and the SHA-256 of option_probs.parquet, which is not in git.

Each study turns the raw probabilities into calibrated twins itself.
"""
import argparse
import gc
import json
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from sdt.data.clean import OUT_PATH, OUTCOMES
from sdt.experiment import git, in_experiments, require_clean_tree, sha256
from sdt.features import FEATURE_SETS
from sdt.paths import EXPERIMENTS, RAW_CSV
from sdt.twin import readout
from sdt.twin.prompts import RAW, messages

DEFAULT_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
STAGES = ['base', 'counterfactual', 'rerun', 'bf16']
# Counterfactual state for every month of the first block. In April to July 2024, the president and the last
# election's winner were both Democrats and inflation was 3.2% to 3.5%
SCENARIOS = {
    'pres_R': lambda S: S.assign(pres_R=1, winner_R=1),
    'cpi_9.1': lambda S: S.assign(cpi=9.1, cpi_d1y=9.1 - (S.cpi - S.cpi_d1y)),   # year-earlier value unchanged
}
COUNTERFACTUAL_FEATURE_SET = 'demographics_macro_politics'
RERUN_PER_CELL, RERUN_SEED = 10, 0
KEY = ['stage', 'scenario', 'interviews', 'feature_set', 'item']


def parse_months(text):
    """'202404-202407,202408-202609' -> [(202404, 202407), (202408, 202609)]. Raise unless the blocks are in order and
    do not overlap."""
    blocks = [tuple(int(m) for m in b.split('-')) for b in text.split(',')]
    if any(len(b) != 2 or b[0] > b[1] for b in blocks):
        raise ValueError(f'each block must be FIRST-LAST with FIRST <= LAST: {text}')
    if any(a[1] >= b[0] for a, b in zip(blocks, blocks[1:])):
        raise ValueError(f'blocks must be in order and must not overlap: {text}')
    return blocks


def interview_blocks(df, blocks, limit=None):
    """The clean table's web interviews in each block of months, in the table's order, keyed by the block's name
    'FIRST-LAST'; with `limit`, only the first `limit` of each (smoke tests). Raise if a block has none."""
    out = {}
    for first, last in blocks:
        d = df[(df.web == 1) & df.YYYYMM.between(first, last)]
        if d.empty:
            raise ValueError(f'no web interviews in {first}-{last}')
        out[f'{first}-{last}'] = d.head(limit) if limit else d
    return out


def rerun_sample(d):
    """(feature set, item, interviews) for the rerun stage: RERUN_PER_CELL interviews of `d` per item and feature set,
    drawn without replacement from one generator, so each cell gets its own draw."""
    rng = np.random.default_rng(RERUN_SEED)
    for fs in FEATURE_SETS:
        for item in OUTCOMES:
            yield fs, item, d.iloc[np.sort(rng.choice(len(d), min(RERUN_PER_CELL, len(d)), replace=False))]


class Runner:
    """Runs prompts chunk by chunk into out/chunks/, loading the model only when a chunk still has to be run."""

    def __init__(self, out, model_name, batch_size):
        self.chunks = out / 'chunks'
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.log = self.chunks / 'log.jsonl'
        self.model_name, self.batch_size = model_name, batch_size
        self.loaded = None                         # (dtype, tokenizer, model)

    def model(self, dtype):
        if self.loaded is None or self.loaded[0] != dtype:
            self.loaded = None                     # free the other precision's model before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            self.loaded = (dtype, *readout.load(self.model_name, dtype))
        return self.loaded[1:]

    def chunk(self, stage, scenario, interviews, fs, item, d, dtype=torch.float32, batch_size=None):
        """Run the prompts of interviews `d` unless their chunk is already on disk."""
        path = self.chunks / ('--'.join([stage, scenario, interviews, fs, item]) + '.parquet')
        if path.exists():
            return
        batch_size = batch_size or self.batch_size
        start = time.time()
        chats = messages(d, fs, item, scenario=SCENARIOS.get(scenario))
        raw = readout.option_probs(*self.model(dtype), chats, item, batch_size)
        raw = raw.assign(CASEID=d.CASEID, **dict(zip(KEY, [stage, scenario, interviews, fs, item])))
        raw.to_parquet(path.with_suffix('.tmp'), index=False)
        path.with_suffix('.tmp').rename(path)       # a chunk on disk is always complete
        entry = {'chunk': path.stem, 'prompts': len(d), 'seconds': round(time.time() - start, 2),
                 'batch_size': batch_size, 'dtype': str(dtype).removeprefix('torch.'), 'commit': git('rev-parse', 'HEAD')}
        with open(self.log, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(entry, flush=True)


def run_stages(runner, blocks, stages):
    if 'base' in stages:
        for b, d in blocks.items():
            for fs in FEATURE_SETS:
                for item in OUTCOMES:
                    runner.chunk('base', 'actual', b, fs, item, d)
    first, d = next(iter(blocks.items()))
    if 'counterfactual' in stages:
        for scenario in SCENARIOS:
            for item in OUTCOMES:
                runner.chunk('counterfactual', scenario, first, COUNTERFACTUAL_FEATURE_SET, item, d)
    if 'rerun' in stages:
        for fs, item, sample in rerun_sample(d):
            runner.chunk('rerun', 'actual', first, fs, item, sample, batch_size=1)
    if 'bf16' in stages:
        for fs, item, sample in rerun_sample(d):
            runner.chunk('bf16', 'actual', first, fs, item, sample, dtype=torch.bfloat16, batch_size=1)


def device_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else readout.device()


def run(out_dir, description, blocks, model_name=DEFAULT_MODEL, stages=('base',), batch_size=16, limit=None,
        resume=False):
    """Run the stages into out_dir/<today>-<description>/ (which must not exist), or with `resume`, continue the run in
    out_dir/<description>/. `blocks` is a list of (first, last) months. Writes every output of the module docstring."""
    if in_experiments(out_dir):
        require_clean_tree()
    name = description if resume else f"{time.strftime('%Y-%m-%d')}-{description}"
    out = out_dir / name
    if resume:
        if (out / 'run.json').exists():
            raise RuntimeError(f'{name} is complete; a new run needs a new folder')
        log = out / 'chunks' / 'log.jsonl'
        commits = {json.loads(line)['commit'] for line in open(log)} if log.exists() else set()
        if commits - {git('rev-parse', 'HEAD')}:
            raise RuntimeError(f'{name} was run at commit {commits}; check out that commit to resume it')
    else:
        out.mkdir(parents=True)                     # a run never writes into an earlier run's folder
    start = time.time()
    df = pd.read_parquet(OUT_PATH)
    sets = interview_blocks(df, blocks, limit)
    run_stages(Runner(out, model_name, batch_size), sets, stages)

    raw = pd.concat([pd.read_parquet(p) for p in sorted((out / 'chunks').glob('*.parquet'))], ignore_index=True)
    raw = raw[KEY + ['CASEID'] + RAW]
    raw.to_parquet(out / 'option_probs.parquet', index=False)
    log = pd.read_json(out / 'chunks' / 'log.jsonl', lines=True)
    log['stage'] = log.chunk.str.split('--').str[0]
    per_stage = log.groupby('stage')[['prompts', 'seconds']].sum()
    json.dump({'name': name, 'commit': git('rev-parse', 'HEAD'),
               'data_csv_sha256': sha256(RAW_CSV), 'option_probs_sha256': sha256(out / 'option_probs.parquet'),
               'date': time.strftime('%Y-%m-%d'), 'runtime_seconds_last_session': round(time.time() - start),
               'python': sys.version.split()[0],
               'packages': {p: version(p) for p in ['torch', 'transformers', 'numpy', 'pandas', 'scipy']},
               'model': model_name, 'revision': readout.MODELS[model_name], 'device': device_name(),
               'dtype': 'fp32 (released bf16 weights converted exactly), except the bf16 stage',
               'batch_size': batch_size, 'limit': limit, 'interviews': {s: len(d) for s, d in sets.items()},
               'stages': list(stages), 'scenarios': list(SCENARIOS), 'counterfactual_feature_set': COUNTERFACTUAL_FEATURE_SET,
               'rerun': {'per_cell': RERUN_PER_CELL, 'seed': RERUN_SEED},
               'per_stage': {s: {'prompts': int(r.prompts), 'seconds': round(float(r.seconds)),
                                 'prompts_per_second': round(r.prompts / r.seconds, 2)} for s, r in per_stage.iterrows()}},
              open(out / 'run.json', 'w'), indent=1)
    for p in (out / 'chunks').glob('*.parquet'):    # everything is in option_probs.parquet now
        p.unlink()
    print(json.dumps(json.load(open(out / 'run.json'))['per_stage'], indent=1))


def benchmark(model_name, sizes=(16, 32, 64, 128), n=512):
    """Prompts per second and peak GPU memory per batch size, on the longest prompts (demographics_macro_politics,
    BUS5) of the first `n` web interviews. Nothing is saved."""
    tok, model = readout.load(model_name)
    d = pd.read_parquet(OUT_PATH).pipe(lambda df: df[df.web == 1]).head(n)
    texts = [readout.render(c) for c in messages(d, 'demographics_macro_politics', 'BUS5')]
    ids = readout.option_tokens(tok, 3)
    cuda = torch.cuda.is_available()
    readout.next_token_probs(tok, model, texts[:sizes[0]], ids, sizes[0])          # warm-up
    for size in sizes:
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            readout.next_token_probs(tok, model, texts, ids, size)
        except torch.OutOfMemoryError:
            print(f'batch size {size}: out of memory')
            break
        if cuda:
            torch.cuda.synchronize()
        memory = f', peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB' if cuda else ''
        print(f'batch size {size}: {len(texts) / (time.time() - start):.1f} prompts per second{memory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('study', help='the study the run belongs to: the run goes to experiments/<study>/')
    parser.add_argument('description', help='what the run contains; with --resume, the folder name')
    parser.add_argument('--months', help='comma-separated blocks of survey months, each FIRST-LAST')
    parser.add_argument('--model', default=DEFAULT_MODEL, choices=list(readout.MODELS))
    parser.add_argument('--stages', default='base', help=f'comma-separated, from {STAGES}')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, help='first N interviews of each block (smoke tests)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    a = parser.parse_args()
    if a.benchmark:
        benchmark(a.model)
    else:
        stages = a.stages.split(',')
        if set(stages) - set(STAGES):
            parser.error(f'stages must be in {STAGES}')
        if not a.months:
            parser.error('--months is required')
        run(EXPERIMENTS / a.study, a.description, parse_months(a.months), a.model, [s for s in STAGES if s in stages],
            a.batch_size, a.limit, a.resume)
