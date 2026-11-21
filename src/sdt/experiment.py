"""Bookkeeping shared by every run: the commit that produced it, a clean working tree and file hashes.

Each run writes a new folder experiments/<study>/<YYYY-MM-DD>-<description>/ and records in its run.json the commit
(`git('rev-parse', 'HEAD')`) and the SHA-256 of every input file that is not in git.
"""
import hashlib
import subprocess

from sdt.paths import EXPERIMENTS, ROOT


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def require_clean_tree():
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('commit your changes first: outputs must trace back to a commit')


def in_experiments(path):
    """Whether `path` is inside experiments/, where every run must trace back to a commit."""
    return path.resolve().is_relative_to(EXPERIMENTS.resolve())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
