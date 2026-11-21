import os

from sdt.experiment import available_cpus


def test_available_cpus_is_at_most_the_cpus_the_process_may_run_on():
    n = available_cpus()
    assert isinstance(n, int) and 1 <= n <= os.cpu_count()
