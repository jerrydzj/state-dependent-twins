"""Bookkeeping shared by every run: the studies, the commit that produced a run, a clean working tree and file hashes.

Each run writes a new folder experiments/<study>/<YYYY-MM-DD>-<description>/ and records in its run.json the commit
(`git('rev-parse', 'HEAD')`) and the SHA-256 of data.csv and of any earlier run's files it reads, which are not in git.
"""
import hashlib
import os
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


def available_cpus():
    """The CPUs this process may use: its CPU quota if a container sets one (cgroup v2 or v1), else the CPUs it may
    run on. A container often shows every core of its host, which is far more than its quota."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    for quota_file, period_file in [('/sys/fs/cgroup/cpu.max', None),
                                    ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us')]:
        if os.path.exists(quota_file):
            values = open(quota_file).read().split()
            quota, period = values[0], values[1] if period_file is None else open(period_file).read().strip()
            if quota not in ('max', '-1'):
                return max(1, min(count, int(quota) // int(period)))
    return count
