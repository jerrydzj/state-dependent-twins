"""Write the paper's generated inputs to build/: the numbers that the text quotes, the figures, and environment.txt,
the record of the environment that builds the PDF.

The text quotes every number as \\R{key}, so none is typed by hand, and a key missing here stops the build. The checks
below guard the claims that the text makes in words.
"""
import sys

STARTUP = set(sys.modules)      # loaded before this script ran, such as by the venv's .pth files

import calendar                                                                 # noqa: E402
import json                                                                     # noqa: E402
import platform                                                                 # noqa: E402
import subprocess                                                               # noqa: E402
from importlib.metadata import packages_distributions, version                  # noqa: E402
from pathlib import Path                                                        # noqa: E402

import matplotlib                                                               # noqa: E402

matplotlib.use('pgf')
import matplotlib.pyplot as plt                                                 # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator                    # noqa: E402

from sdt.realtime import RUNS                                                   # noqa: E402

RUN = RUNS / '2026-11-22-benchmark'
BUILD = Path(__file__).parent / 'build'

# Figures are typeset by LaTeX at their natural size, so these follow sdt-paper.cls
TEXTWIDTH = 6.0                                                                 # inches
FIGURE_STYLE = {
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}',
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
}


def tracked(path):
    """path, after checking that git tracks it. The paper reads only tracked files, so it builds from a clone."""
    check = subprocess.run(['git', 'ls-files', '--error-unmatch', path.name], cwd=path.parent, capture_output=True)
    if check.returncode:
        sys.exit(f'{path} is not tracked by git. The paper reads only tracked files: add it to git if it may be '
                 'published, or else read aggregate results that a run saves from it.')
    return path


def num(x, digits):
    """x rounded to `digits` places, with a typeset minus sign."""
    return f'{x:.{digits}f}'.replace('-', r'\textminus{}')


def pct(x):
    return num(100 * x, 0) + r'\%'


def month(yyyymm):
    return f'{calendar.month_name[yyyymm % 100]} {yyyymm // 100}'


def span(first, last):
    """A short month range, such as 'Aug 2024--Sep 2026' or 'Mar--Sep 2026'."""
    start = calendar.month_abbr[first % 100] + ('' if first // 100 == last // 100 else f' {first // 100}')
    return f'{start}--{calendar.month_abbr[last % 100]} {last // 100}'


def primary_values(name, r):
    share = r['as_share_of_avoidable_loss']
    return {f'{name}.T': str(r['T']), f'{name}.first': month(r['first_month']), f'{name}.last': month(r['last_month']),
            f'{name}.span': span(r['first_month'], r['last_month']),
            f'{name}.mean': num(r['mean'], 4), f'{name}.lo': num(r['lo'], 4), f'{name}.hi': num(r['hi'], 4),
            f'{name}.share': pct(share['mean']), f'{name}.share.lo': pct(share['lo']),
            f'{name}.share.hi': pct(share['hi']), f'{name}.stat': num(r['statistic'], 2), f'{name}.df': str(r['df']),
            f'{name}.p': num(r['p'], 2)}


def secondary_values(s):
    show = lambda v, digits: v if isinstance(v, str) else num(v, digits)
    values = {'curve.T': str(len(s['months'])), 'curve.first': month(s['months'][0]),
              'curve.last': month(s['months'][-1])}
    for name, digits in [('rho', 2), ('slope', 5), ('breakeven', 1)]:
        values |= {name: show(s[name]['estimate'], digits), f'{name}.lo': show(s[name]['lo'], digits),
                   f'{name}.hi': show(s[name]['hi'], digits)}
    return values


def numbers_tex(values):
    lines = [f'% Written by inputs.py from {RUN.relative_to(RUNS.parent.parent)}. Do not edit.',
             r'\makeatletter',
             r'\newcommand\R[1]{\ifcsname sdt@R@#1\endcsname\csname sdt@R@#1\endcsname'
             r'\else\PackageError{numbers}{No result named #1}{Add it to inputs.py.}\fi}']
    lines += [rf'\@namedef{{sdt@R@{key}}}{{{value}}}' for key, value in values.items()]
    return '\n'.join(lines + [r'\makeatother', ''])


def curve_figure(s, path):
    h = list(range(1, 13))
    d = [s['curve'][str(k)] for k in h]
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(0.7 * TEXTWIDTH, 2.4), layout='constrained')
        ax.axhline(0, color='0.65', lw=0.6)
        ax.plot(h, d, color='0.1', lw=1, marker='o', ms=3.5)
        ax.set_xticks(h)
        ax.yaxis.set_major_locator(MultipleLocator(0.002))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:.3f}$'))
        ax.set_xlabel(r'Horizon $h$ (months)')
        ax.set_ylabel(r'$\bar d(h)$')
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(path)
        plt.close(fig)


def environment():
    """The Python version, the version of every package this script has loaded, and the TeX Live snapshot."""
    names = packages_distributions()
    loaded = {dist for module in set(sys.modules) - STARTUP for dist in names.get(module.split('.')[0], [])} - {'sdt'}
    folder = Path(__file__).parent
    lines = [f'# Written by papers/{folder.name}/inputs.py: the environment that built {folder.name}.pdf. The packages',
             f'# are those that inputs.py loads; `uv pip install -r papers/{folder.name}/environment.txt` installs them.',
             f'# Python {platform.python_version()}']
    tex = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    root = subprocess.run(['kpsewhich', '-var-value', 'SELFAUTOPARENT'], capture_output=True, text=True).stdout.strip()
    tlpdb = Path(root) / 'tlpkg' / 'texlive.tlpdb'
    if tlpdb.exists():
        revision = max(int(line.split()[1]) for line in tlpdb.read_text().splitlines() if line.startswith('revision '))
        tex += f', packages at TeX Live revision r{revision}'
    lines.append(f'# {tex}')
    lines += sorted(f'{dist}=={version(dist)}' for dist in loaded)
    return '\n'.join(lines + [''])


def main():
    primary = json.loads(tracked(RUN / 'primary.json').read_text())
    secondary = json.loads(tracked(RUN / 'secondary.json').read_text())
    meta = json.loads(tracked(RUN / 'run.json').read_text())

    # Claims the text makes in words
    assert meta['units_with_errors'] == {}
    assert primary['outcome'] == 'neither shown more accurate'
    # A positive d means that the twin pick's loss is the lower
    assert abs(primary['mean'] - (primary['simple_mean_loss'] - primary['twin_mean_loss'])) < 1e-12
    assert sorted(int(h) for h in secondary['curve']) == list(range(1, 13))

    values = {'run': rf'\path{{experiments/realtime/{meta["name"]}}}',
              'twins': rf'\path{{experiments/realtime/{meta["twins"]["name"]}}}'}
    values |= primary_values('primary', primary)
    for name in ['known', 'unseen']:
        values |= primary_values(name, primary['subperiods'][name])
    values |= secondary_values(secondary)

    BUILD.mkdir(exist_ok=True)
    (BUILD / 'numbers.tex').write_text(numbers_tex(values))
    curve_figure(secondary, BUILD / 'curve.pgf')
    (BUILD / 'environment.txt').write_text(environment())


if __name__ == '__main__':
    main()
