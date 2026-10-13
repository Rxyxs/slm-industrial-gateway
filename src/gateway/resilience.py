"""Rate limiting y caché de respuestas.

Ambos funcionan en memoria del proceso y opcionalmente sobre Redis, para que
varias réplicas del gateway en el mismo IPC compartan cupo y caché.

La clave de caché cuantiza cada valor de telemetría en buckets del 1% del rango
normal del tag: dos lecturas que caen en el mismo bucket producen la misma
recomendación y no tiene sentido pagar otra inferencia por eso. Como todo
bucketing, dos lecturas casi iguales a ambos lados de un borde caen en buckets
distintos; el costo es una inferencia de más, nunca una respuesta equivocada.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from typing import Any, Callable

from src.gateway.schemas import InferenceRequest
from src.gateway.tags import TAG_REGISTRY


class TokenBucketLimiter:
    def __init__(self, rate_per_minute: int, burst: int, clock: Callable[[], float] = time.monotonic):
        self.rate_per_s = rate_per_minute / 60.0
        self.burst = float(burst)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client_id: str) -> tuple[bool, float]:
        """(permitido, segundos hasta el próximo token)."""
        now = self._clock()
        tokens, last = self._buckets.get(client_id, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate_per_s)
        if tokens >= 1.0:
            self._buckets[client_id] = (tokens - 1.0, now)
            return True, 0.0
        self._buckets[client_id] = (tokens, now)
        wait = (1.0 - tokens) / self.rate_per_s if self.rate_per_s > 0 else float("inf")
        return False, wait


def cache_key(req: InferenceRequest, context: str = "") -> str:
    parts: list[Any] = [req.asset_id, req.mode, req.operator_note.strip().lower(), context]
    for s in sorted(req.telemetry, key=lambda x: x.tag):
        spec = TAG_REGISTRY[s.tag]
        resolution = (spec.high - spec.low) / 100.0
        parts.append((s.tag, math.floor((s.value - spec.low) / resolution), s.quality))
    blob = json.dumps(parts, separators=(",", ":"), default=str)
    return "slmgw:cache:" + hashlib.sha256(blob.encode()).hexdigest()[:32]


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None or item[0] < self._clock():
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return item[1]

    async def set(self, key: str, value: str) -> None:
        self._data[key] = (self._clock() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


class RedisCache:
    def __init__(self, redis, ttl_s: float):
        self._redis = redis
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value.decode() if isinstance(value, bytes) else value

    async def set(self, key: str, value: str) -> None:
        await self._redis.set(key, value, px=int(self.ttl_s * 1000))
