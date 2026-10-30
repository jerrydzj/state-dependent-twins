"""Run the MVP twins and save the results to a new folder experiments/mvp/<date>-<description>/.

    python -m sdt.mvp.twin_run twin-pilot-training-web --stages base,counterfactual,rerun,bf16 --batch-size 64
    python -m sdt.mvp.twin_run speed-test --benchmark       # prompts per second and GPU memory per batch size
    python -m sdt.mvp.twin_run <folder name> --resume ...   # continue an interrupted run at the same commit

*Running* a prompt means the model computes its option probabilities; *scoring* compares probabilities with answers.
Every interview of an interview set is run for every item, whether or not it answered the item, so no run depends on
an answer; interviews without an answer are left out only when scoring.

Interview sets: `train-web`, the training window's web interviews (April to July 2024), on which the intercept shift
is fitted, so every run includes it; and `eval`, the evaluation window.

Stages, in this order:

1. `base`: every interview of each set, for each item and feature set, at --batch-size.
2. `counterfactual`: the train-web interviews, for each item, on demographics_macro_politics, under each scenario of
   SCENARIOS.
3. `rerun`: 10 train-web prompts per item and feature set, drawn at random (seed 0), each run alone (batch size 1).
   Their gap from the base stage measures what batching changes.
4. `bf16`: the same prompts, each run alone with the weights kept in bf16. Their gap from `rerun` measures what fp32
   changes.

The raw option probabilities are written chunk by chunk (stage x scenario x set x feature set x item) to chunks/, so an
interrupted run can continue where it stopped. When every chunk is done, the run writes:

- option_probs.parquet: the raw probabilities `p1` to `p3` of every prompt run, with its stage, scenario, interview
  set, feature set, item and CASEID (not in git: a derivative of the microdata).
- predictions.parquet: base-stage class probabilities of the interviews that answered each item, in the format of the
  supervised runs, for `twin_<feature set>` (zero-shot) and `twin_shift_<feature set>` (intercept-shifted). Not in git.
- shifts.json: the intercept shift per feature set and item, and DUR's middle share.
- run.json: commit, data hash, model and revision, device, package versions, settings, and prompts and seconds per
  stage.

Everything after the raw probabilities is computed from option_probs.parquet without the model, so changing it later
is a new run that reads this one's file and needs no GPU.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sdt.mvp import twin
from sdt.mvp.clean import CLASSES, OUT_PATH, OUTCOMES
from sdt.mvp.prompts import FEATURE_SETS, messages
from sdt.mvp.run import EXPERIMENTS, git, require_clean_tree
from sdt.mvp.windows import split
from sdt.paths import RAW_CSV

DEFAULT_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
INTERVIEW_SETS = ['train-web', 'eval']
STAGES = ['base', 'counterfactual', 'rerun', 'bf16']
# Counterfactual state for every train-web month, in which the president and the last election's winner are both
# Democrats and inflation is 3.2% to 3.5%
SCENARIOS = {
    'pres_R': lambda S: S.assign(pres_R=1, winner_R=1),
    'cpi_9.1': lambda S: S.assign(cpi=9.1, cpi_d1y=9.1 - (S.cpi - S.cpi_d1y)),   # year-earlier value unchanged
}
COUNTERFACTUAL_FEATURE_SET = 'demographics_macro_politics'
RERUN_PER_CELL, RERUN_SEED = 10, 0
KEY = ['stage', 'scenario', 'interviews', 'feature_set', 'item']


def interview_sets(df, names, limit=None):
    """The clean table's interviews of each named set; with `limit`, only the first `limit` of each (smoke tests)."""
    train, test = split(df)
    sets = {'train-web': train[train.web == 1], 'eval': test}
    return {n: sets[n].head(limit) if limit else sets[n] for n in names}


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
            self.loaded = (dtype, *twin.load(self.model_name, dtype))
        return self.loaded[1:]

    def chunk(self, stage, scenario, interviews, fs, item, d, dtype=torch.float32, batch_size=None):
        """Run the prompts of interviews `d` unless their chunk is already on disk."""
        path = self.chunks / ('--'.join([stage, scenario, interviews, fs, item]) + '.parquet')
        if path.exists():
            return
        batch_size = batch_size or self.batch_size
        start = time.time()
        chats = messages(d, fs, item, scenario=SCENARIOS.get(scenario))
        raw = twin.option_probs(*self.model(dtype), chats, item, batch_size)
        raw = raw.assign(CASEID=d.CASEID, **dict(zip(KEY, [stage, scenario, interviews, fs, item])))
        raw.to_parquet(path.with_suffix('.tmp'), index=False)
        path.with_suffix('.tmp').rename(path)       # a chunk on disk is always complete
        entry = {'chunk': path.stem, 'prompts': len(d), 'seconds': round(time.time() - start, 2),
                 'batch_size': batch_size, 'dtype': str(dtype).removeprefix('torch.'), 'commit': git('rev-parse', 'HEAD')}
        with open(self.log, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(entry, flush=True)


def run_stages(runner, sets, stages):
    if 'base' in stages:
        for s, d in sets.items():
            for fs in FEATURE_SETS:
                for item in OUTCOMES:
                    runner.chunk('base', 'actual', s, fs, item, d)
    d = sets['train-web']
    if 'counterfactual' in stages:
        for scenario in SCENARIOS:
            for item in OUTCOMES:
                runner.chunk('counterfactual', scenario, 'train-web', COUNTERFACTUAL_FEATURE_SET, item, d)
    if 'rerun' in stages:
        for fs, item, sample in rerun_sample(d):
            runner.chunk('rerun', 'actual', 'train-web', fs, item, sample, batch_size=1)
    if 'bf16' in stages:
        for fs, item, sample in rerun_sample(d):
            runner.chunk('bf16', 'actual', 'train-web', fs, item, sample, dtype=torch.bfloat16, batch_size=1)


def dur_mid_share(df):
    """The baseline shares' middle share for DUR: its weighted share among the training window's web interviews."""
    train, _ = split(df)
    d = train[(train.web == 1) & train.DUR.notna()]
    return float(np.average(d.DUR == 'mid', weights=d.WT))


def postprocess(raw, df):
    """Base-stage predictions in the supervised format, and the intercept shifts, from the raw option probabilities."""
    key = df.set_index('CASEID')
    mid_share = dur_mid_share(df)
    preds, shifts = [], {}
    for (fs, item), g in raw[raw.stage == 'base'].groupby(['feature_set', 'item'], sort=False):
        g = g[key.loc[g.CASEID, item].notna().to_numpy()]         # interviews that answered the item
        P = twin.class_probs(g, item, mid_share)
        fit = (g.interviews == 'train-web').to_numpy()
        a = key.loc[g.CASEID[fit]]
        b = twin.fit_shift(P[fit], a[item].astype(str), a.WT)
        shifts.setdefault(fs, {})[item] = {'fav': b['fav'], 'unf': b['unf']}
        for name, Q in [(f'twin_{fs}', P), (f'twin_shift_{fs}', twin.apply_shift(P, b))]:
            preds.append(Q[CLASSES].assign(item=item, CASEID=g.CASEID.to_numpy(), model=name))
    pred = pd.concat(preds, ignore_index=True)[['item', 'CASEID'] + CLASSES + ['model']]
    return pred, {'dur_mid_share': mid_share, 'shifts': shifts}


def device_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else twin.device()


def run(description, model_name=DEFAULT_MODEL, interviews=('train-web',), stages=('base',), batch_size=16,
        limit=None, out_dir=EXPERIMENTS, resume=False):
    """Run the stages into out_dir/<today>-<description>/ (which must not exist), or with `resume`, continue the run in
    out_dir/<description>/."""
    if 'train-web' not in interviews:
        raise ValueError('every run includes train-web, on which the intercept shift is fitted')
    if out_dir.resolve().is_relative_to(EXPERIMENTS.resolve()):
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
    sets = interview_sets(df, interviews, limit)
    run_stages(Runner(out, model_name, batch_size), sets, stages)

    raw = pd.concat([pd.read_parquet(p) for p in sorted((out / 'chunks').glob('*.parquet'))], ignore_index=True)
    raw = raw[KEY + ['CASEID'] + twin.RAW]
    pred, shifts = postprocess(raw, df)
    raw.to_parquet(out / 'option_probs.parquet', index=False)
    pred.to_parquet(out / 'predictions.parquet', index=False)
    json.dump(shifts, open(out / 'shifts.json', 'w'), indent=1)
    log = pd.read_json(out / 'chunks' / 'log.jsonl', lines=True)
    log['stage'] = log.chunk.str.split('--').str[0]
    per_stage = log.groupby('stage')[['prompts', 'seconds']].sum()
    json.dump({'name': name, 'commit': git('rev-parse', 'HEAD'),
               'data_csv_sha256': hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
               'date': time.strftime('%Y-%m-%d'), 'runtime_seconds_last_session': round(time.time() - start),
               'python': sys.version.split()[0],
               'packages': {p: version(p) for p in ['torch', 'transformers', 'numpy', 'pandas', 'scipy']},
               'model': model_name, 'revision': twin.MODELS[model_name], 'device': device_name(),
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
    BUS5). Nothing is saved."""
    tok, model = twin.load(model_name)
    d = interview_sets(pd.read_parquet(OUT_PATH), ['train-web'])['train-web'].head(n)
    texts = [twin.render(c) for c in messages(d, 'demographics_macro_politics', 'BUS5')]
    ids = twin.option_tokens(tok, 3)
    cuda = torch.cuda.is_available()
    twin.next_token_probs(tok, model, texts[:sizes[0]], ids, sizes[0])          # warm-up
    for size in sizes:
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            twin.next_token_probs(tok, model, texts, ids, size)
        except torch.OutOfMemoryError:
            print(f'batch size {size}: out of memory')
            break
        if cuda:
            torch.cuda.synchronize()
        memory = f', peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB' if cuda else ''
        print(f'batch size {size}: {len(texts) / (time.time() - start):.1f} prompts per second{memory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('description', help='what the run contains; with --resume, the folder name')
    parser.add_argument('--model', default=DEFAULT_MODEL, choices=list(twin.MODELS))
    parser.add_argument('--interviews', default='train-web', help=f'comma-separated, from {INTERVIEW_SETS}')
    parser.add_argument('--stages', default='base', help=f'comma-separated, from {STAGES}')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, help='first N interviews of each set (smoke tests)')
    parser.add_argument('--out', type=Path, default=EXPERIMENTS, help='parent folder of the run folder')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    a = parser.parse_args()
    if a.benchmark:
        benchmark(a.model)
    else:
        interviews, stages = a.interviews.split(','), a.stages.split(',')
        if set(interviews) - set(INTERVIEW_SETS) or set(stages) - set(STAGES):
            parser.error(f'interview sets must be in {INTERVIEW_SETS} and stages in {STAGES}')
        run(a.description, a.model, interviews, [s for s in STAGES if s in stages], a.batch_size, a.limit, a.out,
            a.resume)
