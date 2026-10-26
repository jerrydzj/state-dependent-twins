# External and derived data (added 2026-10-18)

Nothing in this folder modifies or replaces the files pulled from the Surveys of Consumers site on 2026-10-15 (the parent `data/` folder).

- `fred/` — macro series downloaded from FRED on 2026-10-18 (latest vintage). Five copyrighted series are not in git.
  See `fred/README.md`.
- `sca/interview_windows.csv` — the table in `../Release and Interview Dates.pdf`, transcribed to CSV with a script
  (2000-01 to 2027-12; one row per survey month: preliminary and final release dates, first and last interview date).
  Use it to date every survey month by its actual interview window (which starts in the last week of the previous
  calendar month) when aligning macro data or news to respondents.

  One correction to the PDF: it gives November 2020 an interview start of 2020-11-23, after that window's end
  (2020-11-21) and its preliminary release (2020-11-13); 2020-11-23 is December's start. The CSV uses 2020-10-27, the
  day after the October window closed, since 235 of the other 313 windows from 2000-01 to 2026-02 start one or two
  days after the previous one ends. A news report of the 13 November release says interviews began on 28 October.
  Any start from 14 October to 3 November gives the same latest CPI and unemployment releases and the same election
  winner (before the 3 November election). `tests/test_data.py` checks that every window starts before it ends and
  before its preliminary release.
