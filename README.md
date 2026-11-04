# State-dependent twins

This project studies LLM digital twins on survey answers that depend on the state of the world. A twin is a language
model given a respondent's attributes and asked how that respondent answers.

Published studies of twins mostly predict trait-like outcomes, such as stable attitudes and personality, where a
person's answer depends mainly on who they are. Many answers worth simulating depend on conditions instead: how people
judge the economy moves with prices, jobs and the party in the White House. The project uses the University of Michigan
Surveys of Consumers, which has interviewed people every month since 1978 and records each respondent's attributes, so
every answer can be paired with the conditions of its month. `docs/motivation.md` explains how the question arose.

The work is organised as studies. Each study tests one hypothesis in this setting, freezes its plan in a
preregistration before the predictions it tests exist, and reports its results in a notebook. The first study is
underway, and more will follow.

## Studies

| Study | Hypothesis | Plan | Status |
|---|---|---|---|
| `mvp-twin` | An LLM twin's accuracy in predicting survey answers, relative to supervised models given the same inputs, increases as the inputs become less like those of the training data. | [Plan](docs/prereg/mvp-twin.md), [OSF](https://osf.io/ahgkw) | Registered 2026-11-03; twin run on the evaluation window in progress |

## Repository

| Path | Contents |
|---|---|
| `notebooks/` | The record of the work, numbered in the order it happened. Each states the decisions of its time, so a later notebook or a preregistration replaces any earlier plan it contradicts |
| `docs/prereg/` | One plan per study, frozen by the git tag `prereg/<name>`, and how to register a plan on OSF |
| `src/sdt/` | The code. Each module's docstring says what it does and how to run it |
| `experiments/` | One folder per run, at `<stage>/<date>-<description>/` |
| `data/` | The source, pull date and terms of every dataset, and the external series that may be redistributed |
| `tests/` | Unit tests |
| `scripts/` | Machine setup, such as a GPU machine for twin runs |

## Reproducing

The code needs Python 3.14 and the survey microdata, which this repository cannot redistribute.

1. **Download the microdata** into `data/` as `data/README.md` describes, and check its SHA-256 against the one there.
2. **Install the package** with uv, a Python package manager: `uv venv --python 3.14 .venv`, then
   `uv pip install -e '.[dev]'`. Add the `twin` extra to run language models.
3. **Build the clean table** with `.venv/bin/python -m sdt.mvp.clean`, and run the tests with
   `.venv/bin/python -m pytest`.
4. **Run a stage** with its module, such as `sdt.mvp.run` for the supervised models or `sdt.mvp.twin_run` for the
   twins. Each module's docstring lists its options. Twin runs need an NVIDIA GPU and access to Llama on Hugging Face,
   and `scripts/gpu_setup.sh` prepares a fresh Linux machine.

Each notebook runs top to bottom on the last commit that changed it, which need not be `main`, and `AGENTS.md` says how
to rerun one there. A notebook that reads per-interview predictions needs its experiment rerun first, because those
files derive from the microdata and are not in git.

## Record

- **Plans:** each study's plan is frozen by its git tag and by its OSF registration, which archives the repository at
  that tag. The plan states what was already known when it was written.
- **Runs:** no experiment folder changes after its run. Its `run.json` records the commit, the SHA-256 of the
  microdata and the package versions. Twin runs also record the SHA-256 of their raw probabilities, which are not in
  git. Rescoring writes a new folder.
- **Corrections:** a commit that changes a reported number is marked with `!`, as in `fix!:`, and says which numbers
  moved. `git log --oneline | grep '!:'` lists them. `AGENTS.md` gives the full commit and notebook conventions.

## Licence and attribution

- **Code:** MIT (`LICENSE`).
- **Survey data:** not included. The Surveys of Consumers license the microdata for research under CC BY-NC-ND,
  without the right to redistribute, and the twin prompts quote the questionnaire under its licence. Required
  attribution: Surveys of Consumers, Survey Research Center, University of Michigan. Copyright © 1946–2026 The Regents
  of the University of Michigan.
- **Other data:** the READMEs under `data/` give each source's terms.
- **Preregistrations:** each OSF registration states its licence and how to cite the study.
