# ALFRED real-time series (pulled 2026-10-26)

Every published version of the four series behind the state variables, downloaded from ALFRED (FRED's archive of past
releases) through the FRED API with `python -m sdt.alfred`. Observations start in January 2014; versions run from
each observation's first publication to the pull date.

| File | Series | Source | Frequency | Observations | Rows |
|---|---|---|---|---|---|
| CPIAUCNS.csv | CPI-U, all items, not seasonally adjusted, index 1982-84 = 100 | BLS | monthly | 153 | 157 |
| UNRATE.csv | Unemployment rate, seasonally adjusted, % | BLS | monthly | 153 | 260 |
| GDPC1.csv | Real GDP, seasonally adjusted annual rate, billions of chained dollars | BEA | quarterly | 50 | 399 |
| DFF.csv | Effective federal funds rate, % (every calendar day; weekends and holidays repeat the last business day) | Federal Reserve | daily | 4,678 | 4,683 |

Columns: `date` is the observation (first day of the month or quarter, or the day); `value` is the value as
published; `realtime_start` and `realtime_end` are the first and last days that value was the published one, with
9999-12-31 for a value still current. A blank `value` is FRED's "." for an observation that was never published.

Checks at pull time:

- **Release dates (spot check).** For 57 observations, the first `realtime_start` equals the first-release date in
  the agency's news-release archive: CPI and unemployment for each month from January 2024 to February 2026 (BLS),
  and GDP for each quarter from 2024Q4 to 2025Q4 (BEA). The fed funds rate, revision dates and earlier months were
  not checked.
- **2025 shutdown.** October 2025 CPI and unemployment were never published and are blank. September 2025 values
  came out late (CPI 2025-10-24, unemployment 2025-11-20), and 2025Q3 GDP on 2025-12-23.
- **Revisions.** CPI-U NSA was revised once: May to August 2016, on 2016-10-18. The unemployment rate is revised
  every January with new seasonal factors, and November 2025 also moved from 4.6 to 4.5 on 2026-01-09. GDP is revised
  with each estimate and each annual update; its base year changed from 2012 to 2017 dollars in 2023, which leaves
  growth rates within one version unchanged. The fed funds rate was revised twice, by 0.01 (January 2015, May 2017).

Public domain (U.S. government data). A later pull adds versions published after 2026-10-26 and should leave every
earlier row unchanged.
