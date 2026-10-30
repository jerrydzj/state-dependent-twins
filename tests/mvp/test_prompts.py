import re

import numpy as np
import pandas as pd
import pytest

from sdt.mvp.clean import CLASSES, OUTCOMES, clean, load_raw
from sdt.mvp.prompts import ATTRIBUTES, FEATURE_SETS, OPTIONS, messages, system_message, user_message
from sdt.mvp.state import VARIABLES
from sdt.mvp.windows import split
from sdt.paths import RAW_CSV
from test_clean import BASE

STATE = dict(cpi=3.0, cpi_d1y=-0.3, unrate=4.1, unrate_d1y=0.5, gdp=2.9, gdp_d1y=1.2, effr=5.33, effr_d1y=0.25,
             pres_R=0, winner_R=1)


def row(**cells):
    return clean(pd.DataFrame([{**BASE, **cells}])).iloc[0]


def test_feature_sets_add_their_blocks_only():
    text = {fs: system_message(row(), STATE, fs) for fs in FEATURE_SETS}
    assert 'statistics' not in text['demographics'] and 'Politics' not in text['demographics']
    assert 'statistics' in text['demographics_macro'] and 'Politics' not in text['demographics_macro']
    assert 'statistics' not in text['demographics_politics'] and 'Politics' in text['demographics_politics']
    assert text['demographics_macro_politics'] == '\n\n'.join(
        text['demographics_macro'].split('\n\n')[:-1] + text['demographics_politics'].split('\n\n')[-2:])


def test_state_text():
    text = system_message(row(), STATE, 'demographics_macro_politics')
    assert '- Inflation (the change in consumer prices over the past 12 months): 3.0%, compared with 3.3% a year earlier' in text
    assert '- Unemployment rate: 4.1%, compared with 3.6% a year earlier' in text
    assert '- Federal funds interest rate: 5.33%, compared with 5.08% a year earlier' in text
    assert '- Party of the president: Democratic\n- Party that won the most recent presidential election: Republican' in text
    assert '-0.0' not in system_message(row(), STATE | {'gdp': 0.0, 'gdp_d1y': 0.0}, 'demographics_macro')


def test_persona_lists_every_attribute():
    text = system_message(row(), STATE, 'demographics')
    assert text.endswith('About you:\n- Age: 40\n- Sex: female\n- Region: the West\n'
                         '- Marital status: married or living with a partner\n'
                         '- Children under 18 in your household: 0\n'
                         '- Adults 18 or older in your household, counting yourself: 2\n'
                         '- Education: some college, no degree\n'
                         '- Family income: 41st to 60th percentile of US households\n'
                         '- Home: you own your home\n- Investments in the stock market: yes\n'
                         '- Political party: Democrat')


@pytest.mark.parametrize('col, raw, line', [
    ('AGE', 97, '- Age: 97 or older'), ('NUMKID', 5, '- Children under 18 in your household: 5 or more'),
    ('EDUC', 2, '- Education: no high school diploma'), ('REGION', 2, '- Region: the Midwest'),
    ('INVEST', 5, '- Investments in the stock market: no'), ('POLAFF', 3, '- Political party: independent'),
])
def test_persona_value(col, raw, line):
    assert line in system_message(row(**{col: raw}), STATE, 'demographics').split('\n')


@pytest.mark.parametrize('col', ['AGE', 'SEX', 'NUMADT', 'YTL5', 'HOMEOWN', 'POLAFF'])
def test_missing_attribute_is_left_out(col):
    text = system_message(row(**{col: np.nan}), STATE, 'demographics')
    assert f'- {ATTRIBUTES[col][0]}:' not in text
    assert len(text.split('About you:\n')[1].split('\n')) == len(ATTRIBUTES) - 1


@pytest.mark.parametrize('item', OUTCOMES)
def test_options_are_numbered_and_map_to_classes(item):
    lines = user_message(item).split('\n')
    numbered = [l for l in lines if re.match(r'\d\. ', l)]
    assert [l[0] for l in numbered] == [str(k) for k in range(1, len(OPTIONS[item]) + 1)]
    assert set(OPTIONS[item]) <= set(CLASSES) and len(set(OPTIONS[item])) == len(OPTIONS[item])
    assert lines[-1] == 'Answer with the number of your choice only.'


def test_dur_has_no_middle_option():
    assert OPTIONS['DUR'] == ['fav', 'unf']


@pytest.mark.skipif(not RAW_CSV.exists(), reason='data/data.csv not downloaded')
def test_real_data_prompts_carry_no_dates():
    train, _ = split(clean(load_raw()))
    df = train[train.web == 1]
    for fs in FEATURE_SETS:
        chats = messages(df, fs, 'PAGO')
        assert chats.index.equals(df.index)
        for chat in chats:
            text = chat[0]['content']
            assert not re.search(r'\b(19|20)\d\d\b|\bnan\b|<NA>|None', text), text
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    for item in OUTCOMES:
        assert not any(m in user_message(item) for m in months)


# Proper names that could cue knowledge of a specific period: politicians, the survey, events, Fed chairs
NAMES = ['Biden', 'Trump', 'Harris', 'Obama', 'Michigan', 'Surveys of Consumers', 'COVID', 'pandemic', 'tariff',
         'Powell', 'Yellen']


@pytest.mark.skipif(not RAW_CSV.exists(), reason='data/data.csv not downloaded')
def test_real_data_prompts_carry_no_names():
    train, _ = split(clean(load_raw()))
    df = train[train.web == 1]
    pattern = re.compile('|'.join(NAMES), re.IGNORECASE)
    for item in OUTCOMES:
        chats = messages(df, 'demographics_macro_politics', item)   # every other feature set's text is a subset
        assert not any(pattern.search(m['content']) for chat in chats for m in chat)


def test_scenario_changes_the_state_before_writing():
    df = clean(pd.DataFrame([{**BASE, 'YYYYMM': 202407}]))
    flip = lambda S: S.assign(pres_R=1, winner_R=1)
    actual = messages(df, 'demographics_politics', 'BUS12').iloc[0][0]['content']
    flipped = messages(df, 'demographics_politics', 'BUS12', scenario=flip).iloc[0][0]['content']
    assert 'Party of the president: Democratic' in actual and 'Party of the president: Republican' in flipped
