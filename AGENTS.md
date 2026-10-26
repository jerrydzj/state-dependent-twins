# AGENTS.md

Guidance for agents working in this repository.

## Commit messages

Use Conventional Commits syntax with a small set of research types. The history should show at a glance
which commits changed the evidence: data, results, and fixes that moved numbers.

```
type: summary

Optional body.
```

| Type | Use for | Not for |
|---|---|---|
| `data` | Adding or changing input data, or the code that fetches it | Feature engineering in code (`feat`) |
| `feat` | New analysis capability: a model, baseline, prompt, metric, tier | Running it (`exp`) |
| `exp` | An experiment's config and raw outputs: predictions, logprobs, metrics | Write-ups of results (`docs`) |
| `fix` | Correcting a mistake in code, data handling or text | Deliberate changes of method (`feat`) |
| `docs` | Prose: notebooks (including those that report results), READMEs, the spec, the paper | |
| `chore` | Anything a reader of results can ignore: environment, dependencies, `.gitignore`, refactors, formatting | Anything that changes outputs |

- One logical change per commit. A correction is always `fix`. If the change needs two types to describe it, split it.
- Add `!` before the colon (`fix!:`, `feat!:`) when the commit changes any number already reported in a committed
  notebook, write-up or tagged release. The body then says which numbers moved and by about how much, wrapped at 72.
  `git log --oneline | grep '!:'` is the errata list.
- Only `!` commits have a body. Anything else worth keeping goes in the repo: data source and pull date in the data
  folder's README, run metadata in the experiment outputs (which record the commit SHA that produced them).
- Subject: imperative mood ("add", not "added"); lowercase after the colon except identifiers and proper nouns
  (`PX1`, `Llama 3.1`, `FRED`); at most 72 characters; ASCII only; no trailing period; no scopes.
- Commit straight to `main` unless asked to branch. Keep history linear (rebase, don't merge). Annotated tags mark
  snapshots that write-ups cite (`v0.1-mvp`) and frozen predictions (`prereg/<name>`). Git dates are self-reported, so
  also archive `prereg/` tags externally.

```
data: add FRED macro series and SCA interview windows
feat: score twins by next-token logprobs over answer codes
exp: run Llama 3.1 8B twins on tiers A-C
chore: pin vllm and torch
fix!: separate sitting president from election winner

C1 log-loss skill for GOVT goes from below zero to +0.16.
```

## Notebooks

Notebooks don't have to run on `main`. Every commit that changes a notebook leaves it running top to bottom on that
commit's code and data, with its outputs saved, so `git log -1 -- <notebook>` is the commit where it runs. To rerun
it, check out that commit in a worktree and start Python with `-S`, with the worktree's `src/` and the venv's
`site-packages` on `PYTHONPATH`; otherwise the editable install imports `src/` from the main checkout.

## Experiments

Each run writes a new folder `experiments/<stage>/<YYYY-MM-DD>-<description>/` at one commit, recorded in its
`run.json`, and nothing changes the folder afterwards. The description says what the run contains
(`supervised-demographics-state`). Rescoring or refitting is a new run.
