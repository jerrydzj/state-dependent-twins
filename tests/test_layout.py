"""The code layout of AGENTS.md: shared modules never import a study package, and a study package never imports
another study's package."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src' / 'sdt'
SHARED_PACKAGES = {'data', 'twin'}
STUDIES = {'mvp', 'realtime'}
MODULES = sorted(SRC.rglob('*.py'))


def imported(path):
    """Every sdt module or name that `path` imports, as dotted names (`from sdt import features` gives
    sdt.features)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names |= {node.module} | {f'{node.module}.{a.name}' for a in node.names}
    return {n for n in names if n.split('.')[0] == 'sdt'}


def test_every_package_is_shared_or_a_study():
    packages = {p.name for p in SRC.iterdir() if (p / '__init__.py').exists()}
    assert packages <= SHARED_PACKAGES | STUDIES, packages - SHARED_PACKAGES - STUDIES


@pytest.mark.parametrize('path', MODULES, ids=[str(p.relative_to(SRC)) for p in MODULES])
def test_imports_follow_the_layout(path):
    owner = path.relative_to(SRC).parts[0] if len(path.relative_to(SRC).parts) > 1 else None
    studies = {n.split('.')[1] for n in imported(path) if len(n.split('.')) > 1 and n.split('.')[1] in STUDIES}
    allowed = {owner} if owner in STUDIES else set()
    assert studies <= allowed, f'{path.relative_to(SRC)} imports {sorted(studies - allowed)}'
