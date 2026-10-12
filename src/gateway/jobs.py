"""Cola asíncrona de inferencias que no necesitan respuesta en línea.

Reportes de turno, diagnósticos de lotes de telemetría y consultas de un
historian no deberían competir con el lazo de control por el mismo GPU. Van a
esta cola y los consume un pool fijo de workers, así la concurrencia hacia el
motor queda acotada sin importar cuántos clientes encolen.

Con `REDIS_URL` la cola vive en Redis (lista + hash con TTL) y sobrevive a un
reinicio del gateway; sin Redis usa `asyncio.Queue` en memoria.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Awaitable, Callable

from src.gateway.schemas import InferenceRequest, InferenceResponse, JobStatus

log = logging.getLogger("slm_gateway.jobs")

Handler = Callable[[InferenceRequest, str], Awaitable[InferenceResponse]]
JOB_TTL_S = 3600


class _BaseJobQueue:
    def __init__(self, handler: Handler, workers: int = 2):
        self._handler = handler
        self._n_workers = workers
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self._n_workers)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _run(self, job_id: str, payload: dict) -> None:
        await self._set(job_id, JobStatus(job_id=job_id, status="running"))
        try:
            req = InferenceRequest.model_validate(payload["request"])
            result = await self._handler(req, payload.get("client_id", "anonymous"))
            await self._set(job_id, JobStatus(job_id=job_id, status="done", result=result))
        except Exception as exc:  # el worker no muere por una solicitud mala
            log.exception("job %s failed", job_id)
            await self._set(job_id, JobStatus(job_id=job_id, status="failed",
                                              error=f"{type(exc).__name__}: {exc}"))


class InMemoryJobQueue(_BaseJobQueue):
    def __init__(self, handler: Handler, workers: int = 2, maxsize: int = 1000):
        super().__init__(handler, workers)
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._status: dict[str, JobStatus] = {}

    async def submit(self, req: InferenceRequest, client_id: str) -> str:
        job_id = uuid.uuid4().hex
        self._q.put_nowait((job_id, {"request": req.model_dump(mode="json"), "client_id": client_id}))
        self._status[job_id] = JobStatus(job_id=job_id, status="queued")
        return job_id

    async def get(self, job_id: str) -> JobStatus | None:
        return self._status.get(job_id)

    async def _set(self, job_id: str, status: JobStatus) -> None:
        self._status[job_id] = status

    async def _worker(self, i: int) -> None:
        while True:
            job_id, payload = await self._q.get()
            try:
                await self._run(job_id, payload)
            finally:
                self._q.task_done()

    async def join(self) -> None:
        await self._q.join()


class RedisJobQueue(_BaseJobQueue):
    QUEUE = "slmgw:jobs"

    def __init__(self, redis, handler: Handler, workers: int = 2):
        super().__init__(handler, workers)
        self._redis = redis

    async def submit(self, req: InferenceRequest, client_id: str) -> str:
        job_id = uuid.uuid4().hex
        await self._set(job_id, JobStatus(job_id=job_id, status="queued"))
        await self._redis.lpush(self.QUEUE, json.dumps(
            {"job_id": job_id, "request": req.model_dump(mode="json"), "client_id": client_id}
        ))
        return job_id

    async def get(self, job_id: str) -> JobStatus | None:
        raw = await self._redis.get(f"slmgw:job:{job_id}")
        return JobStatus.model_validate_json(raw) if raw else None

    async def _set(self, job_id: str, status: JobStatus) -> None:
        await self._redis.set(f"slmgw:job:{job_id}", status.model_dump_json(), ex=JOB_TTL_S)

    async def _worker(self, i: int) -> None:
        while True:
            item = await self._redis.brpop(self.QUEUE, timeout=1)
            if item is None:
                continue
            payload = json.loads(item[1])
            await self._run(payload["job_id"], payload)
