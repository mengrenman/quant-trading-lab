"""A training run that waits, for tests of what becomes of a fit's worker processes (a spawned process imports it)."""

import os
import time
from pathlib import Path


def waiting(kind, values, settings, seed, burn, span):
    """Say that the run has begun (a file named for this process, in ``LOB_TEST_BEGUN``), and wait."""
    (Path(os.environ["LOB_TEST_BEGUN"]) / str(os.getpid())).touch()
    time.sleep(120)
    return {}
