# Preregistration: MVP twins

Registered as tag `prereg/mvp-twin`, and on OSF, before any twin prediction for an evaluation interview exists.

## 1. Hypothesis

An LLM twin's accuracy in predicting survey answers, relative to supervised models given the same inputs, increases as
the inputs become less like those of the training data.

The inputs are respondent attributes and the state of the economy and the presidency. The supervised models learn how
answers depend on the inputs only from the training data, whereas the twin brings general knowledge of how people
respond to conditions. The twin is almost certainly worse at how this survey's form turns opinions into answers. If the
form works the same way in every month, that handicap is constant, so the hypothesis concerns the trend of the twin's
relative accuracy, not its level.

## 2. Known before registration

Every evaluation answer and every supervised model's evaluation score are known, as are the twin's predictions for the
training window's web interviews. No twin prediction for an evaluation interview exists, so the test is blind on the
twin's side only.

## 3. Design

- **Data:** the Michigan Surveys of Consumers microdata (`data/data.csv`, SHA-256 `5e0a7e1c...ca22`). The training
  window runs from February 2017 to July 2024. The evaluation window runs from August 2024 to February 2026: 19 months
  and 21,183 interviews, all web. The answers are the five items of the Index of Consumer Sentiment, each coded
  favourable, middle or unfavourable.
- **Inputs:** 11 respondent attributes and 10 state variables: inflation, unemployment, GDP growth and the federal
  funds rate, each as its latest value and its change over a year, and the parties of the president and of the latest
  election winner, as published before each month's interviews began.
- **Twin:** Llama 3.1 8B Instruct (revision `0e9e39f2...`, fp32), given the inputs as text and read from its next-token
  probabilities over the answer options. The shifted twin adds a fitted constant to the log-odds of each answer, so that
  its answer shares match those of the training window's 2,257 web interviews.
- **Comparators:** a multinomial logit on the same inputs, and the baseline shares, which predict those interviews'
  answer shares for everyone. Their predictions are frozen in `experiments/mvp/2026-10-26-supervised-macro-politics-ridge/`.

All three models reproduce the answer shares of the training window's web interviews exactly, so each misses a later
month's shares only through its predicted change since then.

## 4. Primary test

- **Dissimilarity d_t:** the Euclidean distance from month t's state variables to those of the nearest of the 90
  training months, with each variable standardised by its mean and standard deviation over the training months. The
  state is the part of the inputs shared by every interview in a month. d_t peaks in December 2024 and January 2025,
  when a Republican had won the election and a Democrat still governed, and falls across 2025, so it does not track
  elapsed time.
- **Advantage A_t:** a comparator's distributional score in month t minus the shifted twin's, averaged over the five
  items. The distributional score is the Wasserstein distance between the predicted and actual weighted answer shares
  of all respondents: half the gap in the favourable share plus half the gap in the unfavourable share.
- **Statistic:** the Spearman correlation ρ between d_t and A_t across the 19 months, with a 95% interval from a paired
  bootstrap that redraws respondents within months (1,000 draws, seed 0). The interval counts respondent sampling, not
  which months occurred.
- **Outcome:** support if ρ's interval lies above 0 against both comparators, contradiction if it lies below 0 against
  either, and inconclusive otherwise.

The outcome rests on the trend, so the twin need not beat a comparator in any month. Support must hold against both
comparators. Because the logit's errors grow in unfamiliar months, a twin that ignored the state could pass against it.
Against the baseline shares, whose predictions such a twin would nearly copy, it could not.

## 5. Secondary analyses

These are reported in full and do not bear on the outcome. Any other analysis is labelled exploratory.

- **Periods:** Baseline (August to October 2024), Flip (November 2024 to January 2025) and Tariffs (from February 2025).
- **Models:** the zero-shot twin, a scaled twin that also multiplies the log-odds by a fitted factor, other inputs and
  other supervised models.
- **Scores and groups:** individual log loss, each item and each party.
- **Diagnostics:** dissimilarity against elapsed time, and the stability of the calibration across periods.

## 6. Procedure

The tag freezes this plan, the data, the prompts, the model and the calibration. The twin's evaluation predictions are
produced at the tagged commit and committed with their SHA-256 before any score is computed. The report lists every
deviation from this plan.
