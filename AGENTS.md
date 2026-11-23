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
  snapshots that write-ups cite (`v0.1-mvp`), paper releases (`paper/<name>/<YYYY-MM-DD>`) and frozen predictions
  (`prereg/<name>`). Git dates are self-reported, so also register each `prereg/` tag on OSF, following
  `docs/prereg/README.md`.

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
it, check out that commit in a worktree, put the data release it used at `data/data.csv` there (`data/README.md`), build
the clean table there, copy in the run files it reads that are not in git, and start Python with `-S`, with the
worktree's `src/` and the venv's `site-packages` on `PYTHONPATH`; otherwise the editable install imports `src/` from
the main checkout.

A notebook is read once, top to bottom, by someone who has not seen the work. 03's tuning paragraph (section 3.2) and
04's first results paragraph (section 5.1) are models. Write by these rules:

- **Parallel ideas, parallel form.** Give each concept one name, never a synonym for variety. Give parallel parts one
  structure, such as "On the individual score, … On the distributional score, …".
- **One idea per paragraph, without exception.** Every sentence develops that idea, so a sentence about anything else
  moves to the paragraph it belongs to or is cut.
- **Choose each conjunction for its logic.** "And" only adds, whereas "so", "because", "but", "although" and "if" say
  how ideas relate. Never join contrasting or unrelated clauses with "and" or "while".
- **In results, interpret the figure; don't recite it.** Say what the patterns mean for the question instead of listing
  cell values, which the reader can read off the figure. Everywhere else, give exact numbers.
- **Describe a process as precise, ordered steps.** Each step says what is done, to what and with which numbers, in the
  order it happens, instead of compressing several into one claim ("noise can only increase a distance").
- **Write like a person, not a model.** Make the actor the subject and the action the verb. Cut any phrase that carries
  no information, such as an abstract label for something concrete ("Two views show both scores"), a hedge or filler.

## Papers

`papers/` holds the shared design (`sdt-paper.cls`, `sdt.mplstyle`), references (`refs.bib`, `acl_natbib.bst`) and
`render`, plus one folder per paper, such as `papers/realtime-benchmark/`, with its text `<name>.tex` and `inputs.py`.
`inputs.py` writes every quoted number, as `\R{key}` (an unknown key stops the build), and every figure to `build/`. It
reads only files tracked by git, and stops on any other, so a paper builds from a clone. An analysis that needs the
microdata or the twin probabilities runs as an experiment that saves aggregates for `inputs.py`. Papers build with
pdflatex, because arXiv has no LuaLaTeX. Every `refs.bib` entry comes from its DOI (`curl -LH "Accept:
application/x-bibtex" https://doi.org/<doi>`) or arXiv record, never from memory. A table or figure spans the text width
(a table with `tabular*`) or the text width over the golden ratio, with a caption of the same width. A figure loads
`sdt.mplstyle`, where color C0 is the twins and C1 the simple models.

`papers/render <name> [--watch]` builds a draft in `build/` (ignored), stamped with its date and commit. Readers see
only releases, named by date and made when a number, claim or figure changes or a version is posted to SSRN or arXiv:

1. Commit the sources and run `papers/render <name> --release <YYYY-MM-DD>` (it refuses uncommitted sources). It stamps
   "This version: <date>", copies the PDF and `environment.txt` (exact Python, package and TeX Live versions) into the
   paper's folder, and writes the arXiv bundle `build/arxiv.tar.gz`, checked to compile on its own.
2. Commit those two files alone (`docs: release the <name> paper of <YYYY-MM-DD>`) and tag the commit
   `paper/<name>/<YYYY-MM-DD>` (annotated).
3. Publish a GitHub Release of the tag with the PDF and a changelog: which reported numbers moved and by about how
   much, and where the version was posted.

Rerun at its tag in the recorded environment, a release rebuilds byte for byte.

## Code

`src/sdt/` holds shared modules and one package per study. The shared modules (`paths.py`, `experiment.py`, `data/`,
`features.py`, `models.py`, `tune.py`, `twin/`) hold what every study uses, and they never import a study's package. A
study's package (`mvp/`, `realtime/`, listed in `experiment.STUDIES`) holds every choice of its preregistration, such
as its windows, candidates and tests, and it never imports another study's package. A command that runs for a study
writes to `experiments/<study>/`, so either its module or its arguments name the study. The tests of each module sit at
the matching path in `tests/` (`tests/twin/test_run.py` for `src/sdt/twin/run.py`), and `tests/test_layout.py` checks
the import rules.

## Experiments

Each run writes a new folder `experiments/<study>/<YYYY-MM-DD>-<description>/` at one commit, recorded in its
`run.json`, and nothing changes the folder afterwards. The description says what the run contains
(`supervised-demographics-state`). Rescoring or refitting is a new run.
