import pandas as pd

from sdt.paths import DATA


def test_interview_windows_are_ordered():
    # Catches transcription errors such as the November 2020 start date in the source PDF (see data/external/README.md)
    win = pd.read_csv(DATA / 'external' / 'sca' / 'interview_windows.csv',
                      parse_dates=['prelim_release', 'final_release', 'interview_start', 'interview_end'])
    assert win.YYYYMM.is_unique and win.notna().all().all()
    assert (win.interview_start < win.interview_end).all()
    assert (win.interview_start < win.prelim_release).all()
    assert (win.prelim_release < win.final_release).all()
