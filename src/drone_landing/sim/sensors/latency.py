"""A fixed-latency FIFO buffer to simulate sensor / processing delay."""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class LatencyBuffer(Generic[T]):
    """Delays values by a fixed number of steps.

    ``push`` a fresh value each step and receive the value from ``delay_steps`` ago (or ``None``
    until the buffer has filled). Used for camera frames, GPS fixes, etc.
    """

    def __init__(self, delay_steps: int):
        self.delay_steps = max(0, int(delay_steps))
        self._buf: deque[T] = deque(maxlen=self.delay_steps + 1)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, value: T) -> T | None:
        self._buf.append(value)
        if len(self._buf) <= self.delay_steps:
            return None
        return self._buf[0]
