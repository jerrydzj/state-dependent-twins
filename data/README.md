# Data

## Surveys of Consumers (not in git)

The microdata come from the University of Michigan Surveys of Consumers data site, https://data.sca.isr.umich.edu
(the Cross-Section Archive, `/sda.php`), downloaded with all 118 variables as a CSV file with its codebook and setup
files. They are not in git: the site's usage agreement (https://data.sca.isr.umich.edu/agreement.php) licenses the data
CC BY-NC-ND for research use, without the right to redistribute. The code reads only `data.csv`; download it into this
folder to run it.

The archive grows as months are released, and it occasionally revises past months (see `Data Updates and
Revisions.pdf`), so each download is a release, identified by the SHA-256 of its `data.csv`. Every run records that
hash in its `run.json`, and the scoring commands stop if `data.csv` is a different release. The current release has
the plain file names below; an earlier release keeps its files with its download date as a suffix, as in
`data-2026-10-15.csv`. To rerun a run on its release, check out its commit in a worktree and copy that release's file
there as `data/data.csv`.

| Release | `data.csv` | Used by |
|---|---|---|
| 2026-11-21 (plain names) | 351,096 rows, survey months 197801–202609; 146,407,768 bytes, SHA-256 `cdfef6f86b23b9c73966a2c59e4c5ab015bb2dc5e0bde29a163f373f38b331b6` | `prereg/realtime-benchmark` |
| 2026-10-15 (suffix `-2026-10-15`) | 342,345 rows, survey months 197801–202602; 143,100,946 bytes, SHA-256 `5e0a7e1cd2a04e8378e43bc1b4316b4bb5387241bb40744f0c50c6f7cd49ca22` | `prereg/mvp-twin`, and the realtime run's check that its twin probabilities reproduce those of `prereg/mvp-twin` |

| File | Contents |
|---|---|
| `data.csv` | Respondent-level microdata, one row per interview, 118 columns |
| `CODEBOOK.txt` | Variable labels and value codes |
| `DDI XML.txt` | DDI codebook; the only source of labels for the open-ended reason codes |
| `SAS.txt`, `SPSS.txt`, `STATA.txt`, `SDA DDL.txt` | Setup files generated with the CSV download |

The 2026-11-21 release adds the 8,751 interviews of March to September 2026 (the archive's update of 2026-11-12). Of
the interviews in both releases, it changes five values: `NEWS2` for 1 interview of 202405, `HOMEAMT` for 1 interview
of 202408, and `NUMADT` for 3 interviews of 202408 to 202410, which were blank and now hold 7. The archive revised
these "to correctly reflect internal topcoding", and its codebook now labels 5 as "5 or more" with valid codes 1 to 5
and missing codes from 9, so the cleaning (`src/sdt/data/clean.py`) reads 7 as 5 or more. In the codebook entries of
the other variables the studies read, only page and column positions changed, apart from the wording of `DATEPR`'s
label ("data month" for "date" of the previous interview).

Documentation PDFs from the same site, downloaded on 2026-10-15 unless stated:

- `Technical Documentation for the  2024 Methodological Transition to Web Surveys.pdf` (two spaces after "the")
- `Methodological Improvements Begin with April 2024 Preliminary Release.pdf`
- `Comparisons of Variables Collected by Phone and Web.pdf`
- `Phone Questionnaire.pdf`, `Web Questionnaire.pdf`: the site keeps only the current version of each, with no history.
  The phone file is the 2015 core questionnaire (PDF modified 2024-02-29). The web file was created 2024-08-20; the site
  replaced it later on 2026-10-15 with a version that adds a licence page, and the twin prompts' wording
  (`src/sdt/twin/prompts.py`) is identical in both
- `Time Series Codebook.pdf`, `Index Calculation.pdf`, `Procedure to Estimate Price Expectations.pdf`
- `Release and Interview Dates.pdf`, `Historical Prelim:Final Release Dates.pdf`
- `Data Updates and Revisions.pdf`: the version of 2026-11-22 (https://data.sca.isr.umich.edu/fetchdoc.php?docid=80741),
  which covers the 2026-11-12 update; the version downloaded on 2026-10-15 is `Data Updates and
  Revisions-2026-10-15.pdf`
- `Current versus Pre-Pandemic Long-Run Inflation Expectations.pdf`,
  `January 2026 Update- Current versus Pre-Pandemic Long-Run Inflation Expectations.pdf`
- `States by Region.pdf`

Required attribution: Surveys of Consumers, Survey Research Center, University of Michigan.
Copyright © 1946–2026 The Regents of the University of Michigan.

## Added data

`external/` holds macro series and tables added for this project. See `external/README.md`.

## Processed data (not in git)

`processed/` holds tables rebuilt from the files above by code. Never edit them by hand.

| File | Built by | Contents |
|---|---|---|
| `clean.parquet` | `python -m sdt.data.clean` | The columns the studies use, cleaned as described in section 1 of `notebooks/03_mvp_first_run.ipynb`, Feb 2017 on |
| `state.csv` | `python -m sdt.data.state` | The 10 state variables per survey month, with each month's cutoff and source observations (for inspection; models call `state.build`) |
