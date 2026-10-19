# FRED macro series (pulled 2026-10-18)

Downloaded with `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` (latest vintage, not real-time vintages).
Each file has two columns: `observation_date` and the series value. Monthly series are dated on the first of the month.

| File | Series | Frequency | Use in this project |
|---|---|---|---|
| CPIAUCSL.csv | CPI-U, all items, SA | monthly | headline inflation (y/y) as a macro condition |
| CPILFESL.csv | CPI-U less food and energy, SA | monthly | core inflation |
| UNRATE.csv | Civilian unemployment rate | monthly | labour-market condition |
| PAYEMS.csv | Nonfarm payrolls | monthly | labour-market momentum |
| FEDFUNDS.csv | Effective federal funds rate | monthly | monetary-policy stance |
| MORTGAGE30US.csv | 30-year fixed mortgage rate (Freddie Mac) | weekly | housing-credit condition |
| APU000074714.csv | Average price, regular unleaded gasoline | monthly | salient price (1976+) |
| GASREGW.csv | Regular gasoline price | weekly | salient price, finer timing (1990+) |
| SP500.csv | S&P 500 index | daily | equity market (FRED keeps only the last 10 years) |
| NASDAQCOM.csv | Nasdaq Composite index | daily | equity market with full history (1971+) |
| USREC.csv | NBER recession indicator | monthly | regime label |
| USEPUINDXM.csv | Economic Policy Uncertainty index (Baker, Bloom, Davis) | monthly | non-LLM news-based feature |
| EPUTRADE.csv | Categorical EPU: trade policy | monthly | non-LLM news-based feature (tariff episodes) |
| UMCSENT.csv | Official Index of Consumer Sentiment | monthly | validation only, never a feature |
| MICH.csv | Official median 1-year inflation expectation | monthly | validation only, never a feature |

Five series are copyrighted by their sources and are not in git: SP500 and NASDAQCOM (FRED: "Copyrighted: Pre-Approval
Required"), UMCSENT and MICH (University of Michigan) and MORTGAGE30US (Freddie Mac). Re-download them with

    for s in SP500 NASDAQCOM UMCSENT MICH MORTGAGE30US; do
      curl -s -o "$s.csv" "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$s"; done

A later download is a later vintage, so it can differ from the 2026-10-18 pull.

Caveats: these are revised, latest-vintage values. Respondents and models at interview time saw first releases with a publication lag
(e.g., CPI for month m is published mid m+1). Features must therefore be lagged to what was public during each survey's interview window
(see `Release and Interview Dates.pdf`). ALFRED real-time vintages are the stricter alternative.
UMCSENT and MICH extend past the microdata (which end in February 2026), so they also support aggregate-only checks for later months.
