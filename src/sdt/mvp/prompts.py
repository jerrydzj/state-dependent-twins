"""Twin prompts for the MVP: what the twin is told about one interview and one item, as chat messages.

The system message introduces the survey, gives the state variables of the feature set (none for `demographics`) and
describes the respondent. The user message is the item's question and answer options, worded as in the 2024 web
questionnaire. Each item gets its own prompt, so no answer depends on another. Options are numbered 1 to k in the web
form's order, and OPTIONS maps each number back to a clean class. A missing attribute is left out of the description.
No prompt contains a date, a name or a survey month. Putting the messages into a model's chat format is done in
twin.py.

The question wording is used under the questionnaire's CC-BY-NC-ND licence: Surveys of Consumers, Survey Research
Center, University of Michigan. Copyright © 1946-2026 The Regents of the University of Michigan.
"""
import pandas as pd

from sdt.mvp import encode
from sdt.mvp.state import MACRO, POLITICS, VARIABLES

FEATURE_SETS = {'demographics': [], 'demographics_macro': MACRO, 'demographics_politics': POLITICS,
                'demographics_macro_politics': VARIABLES}

INTRO = """You are an adult in the United States taking part in an online survey about the economy.

The questions in the survey will ask you about your family. By family, we mean individuals with whom you combine your \
incomes and live together in the same household. If you do not combine incomes, or do not live together as a family, \
please answer these questions thinking only about your own situation."""

# Respondent attribute -> (label, clean value -> text), in the order they are listed. Labels follow the codebook.
COUNT = lambda n: '5 or more' if n == 5 else str(n)       # NUMKID and NUMADT stop at 5
ATTRIBUTES = {
    'AGE': ('Age', lambda a: '97 or older' if a == 97 else str(a)),
    'SEX': ('Sex', {'male': 'male', 'female': 'female'}.__getitem__),
    'REGION': ('Region', {'W': 'the West', 'NC': 'the Midwest', 'NE': 'the Northeast', 'S': 'the South'}.__getitem__),
    'MARRY': ('Marital status', {'married': 'married or living with a partner', 'divorced': 'divorced or separated',
                                 'widowed': 'widowed', 'never': 'never married'}.__getitem__),
    'NUMKID': ('Children under 18 in your household', COUNT),
    'NUMADT': ('Adults 18 or older in your household, counting yourself', COUNT),
    'EDUC': ('Education', {1: 'no high school diploma', 2: 'high school diploma', 3: 'some college, no degree',
                           4: 'college degree', 5: 'postgraduate degree'}.__getitem__),
    'YTL5': ('Family income', {1: 'bottom 20% of US households', 2: '21st to 40th percentile of US households',
                                  3: '41st to 60th percentile of US households',
                                  4: '61st to 80th percentile of US households', 5: 'top 20% of US households'}.__getitem__),
    'HOMEOWN': ('Home', {'own': 'you own your home', 'rent': 'you rent your home'}.__getitem__),
    'INVEST': ('Investments in the stock market', {1: 'yes', 0: 'no'}.__getitem__),
    'POLAFF': ('Political party', {'Rep': 'Republican', 'Dem': 'Democrat', 'Ind': 'independent'}.__getitem__),
}

# Macro variable -> (label, decimals). Each line gives the latest value and the value a year earlier, which carries
# the same information as the supervised models' level and change.
ECONOMY = {
    'cpi': ('Inflation (the change in consumer prices over the past 12 months)', 1),
    'unrate': ('Unemployment rate', 1),
    'gdp': ('Economic growth (the change in real GDP over the past four quarters)', 1),
    'effr': ('Federal funds interest rate', 2),
}
PARTY = {1: 'Republican', 0: 'Democratic'}

# Item -> (question, options in the web form's order as (clean class, text)). Verbatim from the 2024 web
# questionnaire; the web form shows DUR without a middle option.
QUESTIONS = {
    'PAGO': ('We are interested in how people are getting along financially these days. Would you say that you (and '
             'your family living there) are better off or worse off financially than you were a year ago?',
             [('fav', 'Better now financially than a year ago'), ('mid', 'Same'),
              ('unf', 'Worse now financially than a year ago')]),
    'PEXP': ('Now looking ahead--do you think that a year from now you (and your family living there) will be better '
             'off financially, worse off financially, or just about the same as now?',
             [('fav', 'Will be better off financially a year from now'), ('mid', 'Same'),
              ('unf', 'Will be worse off financially a year from now')]),
    'BUS12': ('Now turning to business conditions in the country as a whole--do you think that during the next 12 '
              "months we'll have good times financially or bad times financially?",
              [('fav', 'Good times financially during the next 12 months'), ('mid', 'Partly good and partly bad'),
               ('unf', 'Bad times financially during the next 12 months')]),
    'BUS5': ("Looking ahead, which would you say is more likely -- that in the country as a whole we'll have "
             'continuous good times during the next 5 years or so, or that we will have periods of widespread '
             'unemployment or depression?',
             [('fav', 'Continuous good times during the next 5 years or so'), ('mid', 'Partly good and partly bad'),
              ('unf', 'Continuous bad times (e.g. periods of widespread unemployment or depression) during the next '
                      '5 years or so')]),
    'DUR': ('Think about the big things people buy for their homes -- such as furniture, a refrigerator, stove, '
            'television, and things like that. Generally speaking, do you think now is a good or a bad time for '
            'people to buy major household items?',
            [('fav', 'Good time to buy major household items'), ('unf', 'Bad time to buy major household items')]),
}
OPTIONS = {item: [c for c, _ in options] for item, (_, options) in QUESTIONS.items()}   # option k + 1 -> class
ANSWER = 'Answer with the number of your choice only.'


def number(x, decimals):
    return f'{round(x, decimals) + 0.0:.{decimals}f}'    # + 0.0 turns -0.0 into 0.0


def economy(s):
    lines = [f'- {label}: {number(s[v], d)}%, compared with {number(s[v] - s[f"{v}_d1y"], d)}% a year earlier'
             for v, (label, d) in ECONOMY.items()]
    return 'The latest official statistics:\n' + '\n'.join(lines)


def politics(s):
    return (f'Politics:\n- Party of the president: {PARTY[s["pres_R"]]}\n'
            f'- Party that won the most recent presidential election: {PARTY[s["winner_R"]]}')


def persona(row):
    lines = [f'- {label}: {text(row[c])}' for c, (label, text) in ATTRIBUTES.items() if pd.notna(row[c])]
    return 'About you:\n' + '\n'.join(lines)


def system_message(row, s, feature_set):
    """The survey introduction, the feature set's state variables `s` (a mapping) and the respondent `row`."""
    columns = FEATURE_SETS[feature_set]
    blocks = [INTRO]
    if set(MACRO) <= set(columns):
        blocks.append(economy(s))
    if set(POLITICS) <= set(columns):
        blocks.append(politics(s))
    return '\n\n'.join(blocks + [persona(row)])


def user_message(item):
    question, options = QUESTIONS[item]
    return '\n'.join([question, ''] + [f'{k}. {text}' for k, (_, text) in enumerate(options, 1)] + ['', ANSWER])


def messages(df, feature_set, item, scenario=None):
    """One chat (system and user message) per interview of the clean table `df`, indexed like it. `scenario`, if
    given, is a function that changes the state table (one row per survey month) before the prompts are written: a
    counterfactual."""
    S = encode.state_table(tuple(sorted(df.YYYYMM.unique())))
    if scenario is not None:
        S = scenario(S)
    user = user_message(item)
    return pd.Series([[{'role': 'system', 'content': system_message(row, S.loc[row['YYYYMM']], feature_set)},
                       {'role': 'user', 'content': user}] for _, row in df.iterrows()], index=df.index)
