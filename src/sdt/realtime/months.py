"""The calendar of the realtime-benchmark plan (sections 4 to 8), in YYYYMM months.

A fit at data end e uses the answers of every month up to e and forecasts every later month up to e + 12. Selection at
data end o scores each candidate by its forecasts at the operating point, L(c, v - 2, v), over the validation months v.
"""
FIRST_WEB = 202404              # the first month with web interviews, and the first data end
LAST_DATA_END = 202608
LAST_MONTH = 202609             # the last target month
FIRST_VALIDATION = 202406       # FIRST_WEB + OPERATING_HORIZON: the first month v with a fit at data end v - 2
OPERATING_HORIZON = 2
MAX_AHEAD = 12                  # each fit forecasts up to 12 months past its data end
VALIDATION_MONTHS = 6
TUNING_EVERY = 3                # boosting is tuned at the first data end and every third one after it
HORIZONS = list(range(1, MAX_AHEAD + 1))
PRIMARY = (202408, 202609)      # section 7: the 26 target months
KNOWN = (202408, 202602)        # the 19 target months known before registration
UNSEEN = (202603, 202609)       # the 7 unseen target months
CURVE = (202506, 202609)        # section 8: the target months common to every horizon from 1 to 12


def index(m):
    """Months since January of year 0, so that consecutive months differ by 1."""
    return (m // 100) * 12 + m % 100 - 1


def month(i):
    return (i // 12) * 100 + i % 12 + 1


def add(m, k):
    return month(index(m) + k)


def between(first, last):
    """Every month from `first` to `last`, both included."""
    return [month(i) for i in range(index(first), index(last) + 1)]


def data_ends(last_month=LAST_MONTH):
    """The data ends from April 2024 to LAST_DATA_END that precede `last_month`, so each fit forecasts at least one
    month up to `last_month`."""
    return between(FIRST_WEB, min(LAST_DATA_END, add(last_month, -1)))


def targets(e, last_month=LAST_MONTH):
    """The months a fit at data end e forecasts."""
    return between(add(e, 1), min(add(e, MAX_AHEAD), last_month))


def tuning_end(e):
    """The data end at which the boosting used at data end e was tuned."""
    return add(e, -((index(e) - index(FIRST_WEB)) % TUNING_EVERY))


def validation_months(o):
    """The validation months of a pick at data end o: the latest VALIDATION_MONTHS months v with
    FIRST_VALIDATION <= v <= o, or all of them if fewer."""
    return between(max(FIRST_VALIDATION, add(o, 1 - VALIDATION_MONTHS)), o) if o >= FIRST_VALIDATION else []
