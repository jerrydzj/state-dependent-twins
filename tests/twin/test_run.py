import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

from sdt.data.clean import OUT_PATH, OUTCOMES
from sdt.features import FEATURE_SETS, state_table
from sdt.twin.run import SCENARIOS, Runner, interview_blocks, parse_months, rerun_sample

FIRST_WEB_MONTHS = (202404, 202405, 202406, 202407)


def test_scenarios():
    S = state_table(FIRST_WEB_MONTHS)
    flipped = SCENARIOS['pres_R'](S)
    assert (flipped[['pres_R', 'winner_R']] == 1).all(axis=None) and (S[['pres_R', 'winner_R']] == 0).all(axis=None)
    hot = SCENARIOS['cpi_9.1'](S)
    assert (hot.cpi == 9.1).all()
    assert np.allclose(hot.cpi - hot.cpi_d1y, S.cpi - S.cpi_d1y)          # the year-earlier value is unchanged
    assert hot.drop(columns=['cpi', 'cpi_d1y']).equals(S.drop(columns=['cpi', 'cpi_d1y']))


def test_parse_months():
    assert parse_months('202404-202407,202408-202609') == [(202404, 202407), (202408, 202609)]
    assert parse_months('202404-202404') == [(202404, 202404)]
    for bad in ['202407-202404', '202404-202408,202408-202609', '202408-202609,202404-202407', '202404',
                '202404-202413', 'abc']:
        with pytest.raises(ValueError):
            parse_months(bad)


@pytest.fixture(scope='module')
def clean_table():
    if not OUT_PATH.exists():
        pytest.skip('data/processed/clean.parquet not built')
    return pd.read_parquet(OUT_PATH)


def test_blocks_hold_the_web_interviews_of_their_months_in_table_order(clean_table):
    blocks = interview_blocks(clean_table, [(202404, 202407), (202408, 202602)])
    assert list(blocks) == ['202404-202407', '202408-202602']
    first, second = blocks.values()
    assert len(first) == 2257 and tuple(sorted(first.YYYYMM.unique())) == FIRST_WEB_MONTHS and (first.web == 1).all()
    assert first.index.is_monotonic_increasing and second.index.is_monotonic_increasing
    later = clean_table[clean_table.YYYYMM.between(202408, 202602)]
    assert (later.web == 1).all() and second.index.equals(later.index)    # every interview from August 2024 is web
    assert all(len(d) == 5 for d in interview_blocks(clean_table, [(202404, 202407)], limit=5).values())
    with pytest.raises(ValueError, match='no web interviews'):
        interview_blocks(clean_table, [(202301, 202303)])


def test_rerun_sample_draws_ten_per_cell(clean_table):
    d = interview_blocks(clean_table, [(202404, 202407)])['202404-202407']
    cells = list(rerun_sample(d))
    assert len(cells) == len(FEATURE_SETS) * len(OUTCOMES) and all(len(s) == 10 for _, _, s in cells)
    assert len({tuple(s.CASEID) for _, _, s in cells}) == len(cells)               # each cell has its own draw
    assert [tuple(s.CASEID) for _, _, s in rerun_sample(d)] == [tuple(s.CASEID) for _, _, s in cells]


def test_runner_skips_chunks_on_disk(tmp_path, clean_table):
    runner = Runner(tmp_path, 'unused', 8)
    runner.model = lambda dtype: pytest.fail('the model must not be loaded for a chunk on disk')
    (tmp_path / 'chunks' / 'base--actual--202404-202407--demographics--PAGO.parquet').touch()
    runner.chunk('base', 'actual', '202404-202407', 'demographics', 'PAGO', clean_table.head(2))
