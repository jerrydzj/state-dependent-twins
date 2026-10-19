# External and derived data (added 2026-10-18)

Nothing in this folder modifies or replaces the files pulled from the Surveys of Consumers site on 2026-10-15 (the parent `data/` folder).

- `fred/` — macro series downloaded from FRED on 2026-10-18 (latest vintage). Five copyrighted series are not in git.
  See `fred/README.md`.
- `sca/interview_windows.csv` — the table in `../Release and Interview Dates.pdf`, transcribed to CSV with a script
  (2000-01 to 2027-12; one row per survey month: preliminary and final release dates, first and last interview date).
  Use it to date every survey month by its actual interview window (which starts in the last week of the previous
  calendar month) when aligning macro data or news to respondents.
