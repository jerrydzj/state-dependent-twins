"""Bookkeeping shared by every run: the studies, the commit that produced a run, a clean working tree and file hashes.

Each run writes a new folder experiments/<study>/<YYYY-MM-DD>-<description>/ and records in its run.json the commit
(`git('rev-parse', 'HEAD')`) and the SHA-256 of data.csv and of any earlier run's files it reads, which are not in git.
"""
import hashlib
import subprocess

from sdt.paths import EXPERIMENTS, ROOT

STUDIES = ['mvp', 'realtime']       # each has a package sdt.<study> and a folder experiments/<study>/


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
