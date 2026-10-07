from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timed() -> Iterator[callable]:
    start = perf_counter()

    def elapsed() -> float:
        return perf_counter() - start

    yield elapsed
