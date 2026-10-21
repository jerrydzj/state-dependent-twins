# MVP: cleaning and encoding

How the raw Surveys of Consumers columns become model inputs for the MVP in `notebooks/02_mvp_plan.ipynb`. It covers
only the MVP's 5 outcomes and 11 inputs, chosen by that notebook's rules (E1-E4, O1-O3). Later studies with other
columns or data cuts write their own spec and reuse what applies from this one. The MVP code implements this file; when
they disagree, fix one of them in the same commit.

## Two steps

1. **Clean** (once, shared by every model): map raw codes to clean values, set missing codes to NaN, merge rare levels.
   The output is one table, one row per interview.
2. **Encode** (per model family): turn the clean table into model features.

| Column type | MNL | GB |
|---|---|---|
| Ordered (income, education, age, counts) | One-hot, bucketed first if many levels | Integer |
| Unordered | One-hot | Native categorical (`categorical_features`) |
| Binary | 0/1 | 0/1 |
| Missing | Its own level, `NR` | NaN |

MNL one-hot drops one **reference level** per input: the most common one, so each coefficient compares a group with the
largest group. NR is never the reference. Feature columns are fixed by this spec, never inferred from the data.

Why: MNL is linear, so an integer would force equal steps between levels; one-hot lets each level find its own effect.
Trees split on thresholds, so an integer keeps the order without assuming equal steps. Missing values get a level or a
NaN rather than an imputed value, because the models must predict every evaluation respondent, and on web 9-13% leave
demographics blank, mostly not at random.

## Rows

- Survey months Feb 2017 (W1, the first month `POLAFF` is asked every month) to the last archived month.
- Outcome DK/NA rows are dropped **per item**: a respondent missing `BUS5` still counts for the other four items.

## Outcomes

Each is mapped to `fav` / `mid` / `unf`, and DK/NA is dropped.

| Column | fav | mid | unf | Dropped |
|---|---|---|---|---|
| `PAGO` | 1 | 3 | 5 | 8, 9 |
| `PEXP` | 1 | 3 | 5 | 8, 9 |
| `BUS12` | 1, 2 | 3 | 4, 5 | 8, 9 |
| `BUS5` | 1, 2 | 3 | 4, 5 | 98, 99 |
| `DUR` | 1 | 3 | 5 | 8, 9 |

**Mode and the middle class.**
- `BUS12` and `BUS5` show "partly good, partly bad" as an option on web, while phone records it only when someone
  volunteers it: about 4-7% of answers on phone, 36-47% on web.
- The web form for `DUR` offers no middle option at all, so the middle is about 1% on web.
- Codes 2 and 4 ("with qualifications") occur only on phone.
- We keep three classes and let the `web` indicator absorb the shift. In MNL, `web` has one coefficient per class. In GB,
  each class has its own trees, which can split on `web`. Both learn the shift from the Apr-Jul 2024 web interviews
  only.

## Inputs

| Column | Clean values | MNL | GB | Notes |
|---|---|---|---|---|
| `YTL5` | 1-5; blank → NaN | one-hot + NR | integer | Quintile within survey year; ~5% blank in both modes |
| `HOMEOWN` | own (1), rent (2); 99, blank → NaN | one-hot + NR | categorical | |
| `INVEST` | yes (1) → 1; 5 → 0 | 0/1 | 0/1 | 5 pools no, DK and NA; can't be separated |
| `AGE` | 18-97; blank → NaN | buckets 18-34, 35-49, 50-64, 65-74, 75+, NR | integer | 97 = 97 or older; 13% blank on web |
| `REGION` | W, NC, NE, S | one-hot | categorical | Never missing |
| `SEX` | male (1), female (2); blank → NaN | one-hot + NR | categorical | |
| `MARRY` | married (1), divorced/separated (3), widowed (4), never (5); blank → NaN | one-hot + NR | categorical | No code 2 |
| `NUMKID` | 0-5; blank → NaN | buckets 0, 1, 2, 3+, NR | integer | 5 is a top code (5 or more) |
| `NUMADT` | 1-5; 8, 9, blank → NaN | buckets 1, 2, 3, 4+, NR | integer | Web missing is blank to Nov 2024, code 9 after |
| `EDUC` | 1, 2 merged → 1 (no HS); then HS, some college, BA, grad → 2-5; blank → NaN | one-hot + NR | integer | Codes 1-2 are rare, ~1% on web |
| `POLAFF` | Rep (1), Dem (2), Ind (3); 8, 9, blank → NaN | one-hot + NR | categorical | NR kept apart from Ind; feeds party × president |

## Design columns

| Column | Clean | Use |
|---|---|---|
| `CASEID` | as is | Row key |
| `YYYYMM` | as is | Survey month; sets windows and state variables |
| `METHOD` | `web` = 1 if `METHOD` = 2, else 0 | Feature in both families. Both modes ran in Apr-Jun 2024 |
| `WT` | as is, untrimmed | Sample weight for fitting and scoring. Phone 0.5-1.5, web 0.02-3.0 |
| `ID`, `IDPREV`, `DATEPR` | as is | Panel links. Unused in the MVP (reinterviews are independent rows), kept for prior-answer features later |
