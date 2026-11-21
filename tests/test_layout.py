"""The code layout of AGENTS.md: shared modules never import a study package, a study package never imports another
study's package, and modules that calibrate and score twins import without PyTorch."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from sdt.experiment import STUDIES

SRC = Path(__file__).resolve().parents[1] / 'src' / 'sdt'
SHARED_PACKAGES = {'data', 'twin'}
MODULES = sorted(SRC.rglob('*.py'))


def imported(path):
    """Every sdt module or name that `path` imports, as dotted names (`from sdt import features` gives
    sdt.features). Raise on a relative import, which the codebase does not use and this check would miss."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f'{path.name}:{node.lineno}: relative import; use an absolute one')
            names |= {node.module} | {f'{node.module}.{a.name}' for a in node.names}
    return {n for n in names if n.split('.')[0] == 'sdt'}


def test_every_package_is_shared_or_a_study():
    packages = {p.name for p in SRC.iterdir() if (p / '__init__.py').exists()}
    assert packages <= SHARED_PACKAGES | set(STUDIES), packages - SHARED_PACKAGES - set(STUDIES)


@pytest.mark.parametrize('path', MODULES, ids=[str(p.relative_to(SRC)) for p in MODULES])
def test_imports_follow_the_layout(path):
    owner = path.relative_to(SRC).parts[0] if len(path.relative_to(SRC).parts) > 1 else None
    studies = {n.split('.')[1] for n in imported(path) if len(n.split('.')) > 1 and n.split('.')[1] in STUDIES}
    allowed = {owner} if owner in STUDIES else set()
    assert studies <= allowed, f'{path.relative_to(SRC)} imports {sorted(studies - allowed)}'


BLOCK_PYTORCH = """
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'transformers'}:
            raise ImportError(f'{name} is blocked')

sys.meta_path.insert(0, Block())
"""


def test_calibration_and_scoring_import_without_pytorch():
    # A fresh interpreter in which importing torch or transformers fails, as on a machine without the `twin` extra
    code = BLOCK_PYTORCH + 'import sdt.features, sdt.twin.calibration, sdt.mvp.twin_calibration, sdt.mvp.twin_score'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    blocked = subprocess.run([sys.executable, '-c', BLOCK_PYTORCH + 'import sdt.twin.readout'], capture_output=True)
    assert blocked.returncode != 0                                           # the block works
