# Data

## Surveys of Consumers (not in git)

The files directly in this folder were downloaded on 2026-10-15 from the University of Michigan Surveys of Consumers
data site, https://data.sca.isr.umich.edu (microdata from the Cross-Section Archive, `/sda.php`). They are not in git:
the site's usage agreement (https://data.sca.isr.umich.edu/agreement.php) licenses the data CC BY-NC-ND for research
use, without the right to redistribute. The notebooks read only `data.csv`; download it into this folder to run them.

| File | Contents |
|---|---|
| `data.csv` | Respondent-level microdata, one row per interview: 342,345 rows, 118 columns, survey months 197801–202602 |
| `CODEBOOK.txt` | Variable labels and value codes |
| `DDI XML.txt` | DDI codebook; the only source of labels for the open-ended reason codes |
| `SAS.txt`, `SPSS.txt`, `STATA.txt`, `SDA DDL.txt` | Setup files generated with the CSV download |

The 2026-10-15 `data.csv` is 143,100,946 bytes with SHA-256
`5e0a7e1cd2a04e8378e43bc1b4316b4bb5387241bb40744f0c50c6f7cd49ca22`. Later downloads can differ: the archive grows as
months are released and past months are occasionally revised (see `Data Updates and Revisions.pdf`).

Documentation PDFs from the same site:

- `Technical Documentation for the  2024 Methodological Transition to Web Surveys.pdf` (two spaces after "the")
- `Methodological Improvements Begin with April 2024 Preliminary Release.pdf`
- `Comparisons of Variables Collected by Phone and Web.pdf`
- `Phone Questionnaire.pdf`, `Web Questionnaire.pdf`
- `Time Series Codebook.pdf`, `Index Calculation.pdf`, `Procedure to Estimate Price Expectations.pdf`
- `Release and Interview Dates.pdf`, `Historical Prelim:Final Release Dates.pdf`, `Data Updates and Revisions.pdf`
- `Current versus Pre-Pandemic Long-Run Inflation Expectations.pdf`,
  `January 2026 Update- Current versus Pre-Pandemic Long-Run Inflation Expectations.pdf`
- `States by Region.pdf`

Required attribution: Surveys of Consumers, Survey Research Center, University of Michigan.
Copyright © 1946–2026 The Regents of the University of Michigan.

## Added data

`external/` holds macro series and tables added for this project. See `external/README.md`.
