"""Circuit breaker asíncrono por backend.

CLOSED    -> las llamadas pasan; `failure_threshold` fallas seguidas lo abren.
OPEN      -> se rechaza sin llamar durante `reset_timeout_s` (el backend caído
             no consume presupuesto de latencia de la planta).
HALF_OPEN -> pasado el timeout se deja pasar UNA llamada de prueba: si sale bien
             cierra, si falla vuelve a abrir y reinicia el reloj.

El reloj es inyectable para que los tests no dependan de `sleep`.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    def __init__(self, name: str, retry_in_s: float):
        super().__init__(f"circuit '{name}' open, retry in {retry_in_s:.1f}s")
        self.name = name
        self.retry_in_s = retry_in_s


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        reset_timeout_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()
        self.total_failures = 0
        self.total_rejections = 0

    @property
    def state(self) -> BreakerState:
        if (
            self._state == BreakerState.OPEN
            and self._clock() - self._opened_at >= self.reset_timeout_s
        ):
            return BreakerState.HALF_OPEN
        return self._state

    def allows_request(self) -> bool:
        state = self.state
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.HALF_OPEN:
            return not self._probe_in_flight
        return False

    async def _before(self) -> None:
        async with self._lock:
            state = self.state
            if state == BreakerState.OPEN:
                self.total_rejections += 1
                remaining = self.reset_timeout_s - (self._clock() - self._opened_at)
                raise CircuitOpenError(self.name, max(remaining, 0.0))
            if state == BreakerState.HALF_OPEN:
                if self._probe_in_flight:
                    self.total_rejections += 1
                    raise CircuitOpenError(self.name, 0.0)
                self._state = BreakerState.HALF_OPEN
                self._probe_in_flight = True

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._probe_in_flight = False
            self._state = BreakerState.CLOSED

    async def record_failure(self) -> None:
        async with self._lock:
            self.total_failures += 1
            self._probe_in_flight = False
            if self._state == BreakerState.HALF_OPEN:
                self._trip()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._failures = 0

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        await self._before()
        try:
            result = await fn()
        except asyncio.CancelledError:
            # Cancelado por el deadline global: no es culpa del backend en sí,
            # pero tampoco respondió a tiempo. Cuenta como falla.
            await self.record_failure()
            raise
        except Exception:
            await self.record_failure()
            raise
        await self.record_success()
        return result

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._failures,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
        }
