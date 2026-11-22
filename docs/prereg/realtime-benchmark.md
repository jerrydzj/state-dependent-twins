# Preregistration: a real-time benchmark of LLM twins against simple models

Registered as tag `prereg/realtime-benchmark` and on OSF.

## 1. Question

When survey answers depend on external conditions, and a forecast may use the answers and state variables available
when it is made and the attributes of the people interviewed, do LLM twins predict answer shares more accurately than
simple models?

The study answers this question in one setting, the University of Michigan Surveys of Consumers, at the setting's
operating point: forecasts made when the latest available answers are two months old.

The primary test (section 7) is two-sided and states no directional hypothesis. The secondary analysis (section 8)
describes how the comparison changes with the horizon, the age of the latest available answers, where the expectation is
that the twins' advantage rises with the horizon.

## 2. Known before registration

Every answer up to February 2026 is known, as are all results of notebooks 03 to 06 (`prereg/mvp-twin`) and later
exploratory checks on those months, including persistence and twins recalibrated on the previous month's answers, of
which persistence matched or beat the twins. Of March to September 2026, only the published shares of one item, `PAGO`,
were seen during drafting. Apart from the forecasts of notebooks 03 to 06, all from data ending in July 2024, no
forecast of a month from August 2024 on has been made at a horizon of two months or more.

## 3. Data

The data are the Surveys of Consumers microdata. The evaluation window runs from August 2024 to September 2026: 26
months, all interviewed on the web. The items, answer coding, 11 respondent attributes, 10 state variables and their
dating are those of `prereg/mvp-twin`. The twin is Llama 3.1 8B Instruct with the prompts and option-probability readout
of `prereg/mvp-twin`.

## 4. Forecast task

A forecast of month t is made on its forecast date, the day before month t's interview window opens, at a horizon h:
the latest answers it may use are those of month t − h. It uses the state variables as published on the forecast date,
the answers of every month up to month t − h and, although they are not yet public, the attributes of month t's
respondents. Forecasts are therefore conditional on the realized sample: they measure how that month's respondents
answer, not a forecast publishable before fieldwork. When the survey publishes on schedule, month t − 2 is the latest
month whose final shares and microdata are public on the forecast date, so h = 2 is the operating point.

A forecast's predicted shares for an item are its predicted probabilities averaged over the month's web respondents
who answered the item, weighted by `WT`. The actual shares are the weighted shares of their answers. The loss of a
forecast is the squared error of the cumulative shares, (fav − fav̂)² + (unf − unf̂)², averaged over the five items.

## 5. Procedure

The procedure acts as a forecaster would each month: it refits every candidate on the answers then available, and each
class uses the candidate whose recent forecasts were most accurate. It runs in five steps.

1. **Candidates.** A candidate is one model with every setting fixed. Section 6 lists the 32 twin candidates and the 60
   simple-model candidates, which form the two classes.
2. **Fits.** For each data end e, from April 2024 to August 2026, every candidate is fitted on the answers up to month
   e. Fitting estimates the candidate's parameters and tunes its penalties as section 6 describes, without any later
   answer.
3. **Losses.** Each fit forecasts every later month m up to month e + 12, from month m's state variables and
   respondents (section 4). L(c, e, m) is the loss of candidate c, fitted at data end e, on month m. Every L(c, e, m) is
   saved.
4. **Selection.** At each data end o, each class picks one candidate by its recent forecasts at the operating point. A
   candidate's score is its mean L(c, v − 2, v) over the validation months: the six latest months v with
   June 2024 ≤ v ≤ o, or all of them if fewer. Each class picks the candidate with the lowest score, and exact ties go
   to the candidate listed first in section 6. A candidate missing a fit that its score needs is not eligible at o.
5. **Comparison.** For target month t and horizon h, each class's pick at data end o = t − h forecasts month t with its
   fit at o, and d_t(h) = L(simple pick, o, t) − L(twin pick, o, t). A positive d_t(h) means the twin pick is closer to
   the actual shares.

## 6. Candidates

Respondent-level models use interviews from February 2017, and models fitted to web answers use web interviews from
April 2024. A web month is a month with web interviews, and only those interviews count in it. The input sets are D,
D+M, D+P and D+M+P (notebook 04). For breaking ties, candidates are ordered by model as numbered, then by input set as
listed, then by their remaining setting as listed.

**Twins** (32 candidates):

1. **Shifted twin** (16): the zero-shot twin of notebook 05 on each input set, with one constant added to the log-odds
   of the favourable answer and another to those of the unfavourable answer, both against the middle answer. The
   constants are fitted by weighted maximum likelihood to the answers of the latest k web months, for k in
   {1, 3, 6, all}. A k above the number of available web months uses all of them, and candidates that thereby coincide
   count as one. `DUR`'s middle share is its weighted share in the same months.
2. **Scaled twin** (16): the shifted twin with the log-odds also multiplied by a factor, fitted together with the
   constants.

**Simple models** (60 candidates):

3. **Exponential smoothing** (10): for each answer, with s_m its weighted share among month m's web interviews, the
   level follows ℓ_m = α·s_m + (1 − α)·ℓ_(m−1) from ℓ_(April 2024) = s_(April 2024), for α in {1.0, 0.9, ..., 0.1}. The
   forecast is the level at month e, so α = 1 is persistence.
4. **One-stage ridge logit** (12): the ridge logit of notebook 04 on each input set, with each interview's weight
   multiplied by 0.5^(a / half-life), where a is the interview's age in months at month e, so older interviews count
   less, for a half-life in {none, 12, 36}. Its penalty is tuned by notebook 04's month-blocked cross-validation on the
   decayed weights.
5. **Gradient boosting** (12): the boosting of notebooks 03 and 04 on each input set, with the same half-lives. Its
   settings are tuned by notebook 03's month-blocked cross-validation on the decayed weights at April 2024 and every
   third data end after it. At the data ends in between, the model is refitted with the latest tuned settings.
6. **Two-stage ridge** (26), on each input set:
   - **Stage 1** is a multinomial logit with unpenalized intercepts for each month, which are the month's level, an
     unpenalized web indicator, and ridge-penalized coefficients on the 11 respondent attributes and, on input sets
     with P, the four party × president products. The month intercepts absorb the state variables, which enter only
     in stage 2. Because the web indicator is in the model, each month's level is net of mode. The mode effect is
     assumed constant and is identified from April to June 2024, the months with both phone and web interviews. The
     penalty is tuned by 5-fold cross-validation over interviews, so that every month appears in every fold.
   - **Stage 2** forecasts the month's level in one of three ways: exponential smoothing over the levels of web
     months, with the α grid of candidate 3; or, on input sets with M or P, a ridge regression of every month's level
     on the input set's state variables; or, on the same input sets, the level at month e plus the change predicted by
     a ridge regression of every month-to-month change in level on the change in the input set's state variables. Both
     regressions have their penalty tuned by month-blocked cross-validation over months. Smoothing does not use M, so
     its candidates on D+M and D+M+P duplicate those on D and D+P and count once.
   - A respondent's predicted probabilities are the softmax of the forecast level, the web coefficient and the
     respondent's penalized terms.

## 7. Primary test

- **Estimate:** d̄, the mean of d_t(2) over the T = 26 target months, which estimates the expected difference
  E[d_t(2)].
- **Test:** the Diebold–Mariano statistic, d̄ divided by its standard error, with the long-run variance of d_t(2)
  estimated by the Daniell-kernel average of its periodogram over the ⌊T^(1/3)⌋ = 2 lowest Fourier frequencies, and
  compared with a t-distribution with twice that many, 4, degrees of freedom (fixed-m asymptotics; Coroneo and Iacone,
  2020, Journal of Applied Econometrics 35(4)). The test is two-sided at 5%, with critical value 2.776, the 97.5th
  percentile of that t-distribution.
- **Outcome:** the twins are more accurate if p < 0.05 and d̄ > 0, the simple models are more accurate if p < 0.05 and
  d̄ < 0, and neither is shown more accurate otherwise.
- **Interval:** d̄ ± 2.776 × its standard error, the 95% interval of the same test. It is reported in loss units and as
  a share of the simple pick's avoidable loss: d̄ and both ends of the interval divided by the simple pick's mean loss
  minus the mean noise floor. A month's noise floor is the expected loss of a forecast equal to the population's
  shares, estimated for each item and answer as the variance of the weighted share, Σ w_i² (y_i − ŝ)² / (Σ w_i)², and
  averaged over items as the loss is.
- **Subperiods:** d̄ and its interval, computed as above with T the period's number of months, are also reported for
  August 2024 to February 2026, the 19 months known before registration, and for March to September 2026, the 7 unseen
  months. They carry no decision rule.

## 8. Secondary analysis

d̄(h), the mean of d_t(h) over the common target months June 2025 to September 2026, is computed for h = 1 to 12. Three
quantities summarize it:

- **ρ:** the Spearman correlation between h and d̄(h), which gives the curve's direction.
- **Slope:** the least-squares slope of d̄(h) on h, which gives the change per month of horizon.
- **Breakeven:** the smallest h at which the linearly interpolated d̄(h) turns positive, reported as below 1 if
  d̄(1) > 0 and as above 12 if it never does.

All three get 95% intervals from one moving-block bootstrap of target months (blocks of 3, 1,000 draws, seed 0) that
recomputes the whole curve in each draw. ρ and the slope are expected to be positive, meaning that the twins' advantage
rises with the horizon. This expectation carries no decision rule, and the secondary analysis does not
bear on the outcome.

## 9. Exploratory analyses

Exploratory analyses explain the results and look for patterns. They are chosen with the results known, and the report
labels them exploratory. They may include, among others, each model against persistence, the share of change each model
captures, the picks over time, restricted input sets, periods and parties, individual answers, respondent composition,
actual publication calendars and respondent sampling.

## 10. Execution

The tag freezes this plan and the code that implements it. After the tag, the answers and twin probabilities for March
to September 2026 are obtained, and every statistic of sections 7 and 8 is computed once. The report lists every
deviation from this plan.
