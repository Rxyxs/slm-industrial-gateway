"""Enrutador de inferencia edge/cloud con presupuesto de latencia acotado.

Para cada solicitud:
1. Inspecciona la nota del operador; si trae patrones de inyección se descarta
   la nota (la telemetría igual se evalúa) y queda registrado.
2. Busca en caché.
3. Calcula una complejidad en [0, 1] y arma el orden de backends: borde
   primero; nube primero solo si la complejidad supera el umbral y la nube está
   habilitada. Un backend cuyo `max_complexity` es menor que el puntaje se salta.
4. Intenta cada backend con su circuit breaker y con el tiempo que quede del
   deadline global. Error de red, timeout, breaker abierto o salida que no
   cumple el schema -> siguiente backend.
5. Una salida bien formada pero físicamente inválida NO pasa al siguiente
   backend: el modelo entendió la pregunta y propuso algo inseguro; seguir
   probando hasta que alguno lo apruebe sería buscar la respuesta que se quiere.
   Termina en HOLD.
6. Si nada respondió a tiempo: HOLD failsafe, `degraded=True`.

El gateway siempre devuelve una acción dentro del deadline; nunca un 5xx
hacia el SCADA por culpa de un modelo caído.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from src.gateway.audit import AuditLog
from src.gateway.backends import BackendError, InferenceBackend
from src.gateway.breaker import CircuitBreaker, CircuitOpenError
from src.gateway.config import GatewaySettings
from src.gateway.resilience import cache_key
from src.gateway.schemas import InferenceRequest, InferenceResponse, model_output_json_schema
from src.gateway.security import SYSTEM_PROMPT, build_user_prompt, inspect_note
from src.gateway.tags import TAG_REGISTRY
from src.gateway.validation import (
    OutputRejected, check_physics, check_premise, failsafe_hold, parse_action,
)

log = logging.getLogger("slm_gateway.router")

_SCHEMA = model_output_json_schema()


def complexity_score(req: InferenceRequest) -> float:
    """Heurística barata (<10 µs) de qué tan difícil es la consulta.

    Pocas señales, todas en rango y sin nota -> trivial para un SLM de 1-3B.
    Varias variables fuera de rango, sensores dudosos y una nota larga del
    operador -> razonamiento multi-variable donde un modelo grande rinde más.
    """
    n = len(req.telemetry)
    out_of_range = sum(
        1 for s in req.telemetry
        if not (TAG_REGISTRY[s.tag].low <= s.value <= TAG_REGISTRY[s.tag].high)
    )
    not_good = sum(1 for s in req.telemetry if s.quality != "good")
    score = (
        0.25 * min(n / 16, 1.0)
        + 0.35 * min(out_of_range / 3, 1.0)
        + 0.15 * min(not_good / 2, 1.0)
        + 0.25 * min(len(req.operator_note) / 400, 1.0)
    )
    return round(min(score, 1.0), 3)


class InferenceRouter:
    def __init__(
        self,
        settings: GatewaySettings,
        backends: list[InferenceBackend],
        audit: AuditLog,
        cache=None,
        clock=time.monotonic,
    ):
        self.settings = settings
        self.backends = backends
        self.audit = audit
        self.cache = cache
        self._inflight: dict[str, asyncio.Future] = {}
        self._rewarm: dict[str, asyncio.Task] = {}
        self._rewarm_ok_at: dict[str, float] = {}
        self.breakers = {
            b.name: CircuitBreaker(
                b.name, settings.breaker_failure_threshold, settings.breaker_reset_timeout_s, clock
            )
            for b in backends
        }

    def plan(self, score: float) -> list[InferenceBackend]:
        edge = [b for b in self.backends if b.tier == "edge" and score <= b.max_complexity]
        cloud = [b for b in self.backends if b.tier == "cloud"] if self.settings.allow_cloud else []
        if cloud and score > self.settings.cloud_complexity_threshold:
            return cloud + edge
        return edge + cloud

    async def infer(self, req: InferenceRequest, client_id: str = "anonymous",
                    request_id: str | None = None) -> InferenceResponse:
        request_id = request_id or uuid.uuid4().hex
        t0 = time.perf_counter()
        deadline = t0 + self.settings.deadline_s
        elapsed_ms = lambda: round((time.perf_counter() - t0) * 1000, 2)  # noqa: E731

        inspection = inspect_note(req.operator_note)
        note_block = None
        if not inspection.allowed:
            note_block = f"operator_note blocked: {', '.join(inspection.findings)}"
            await self.audit.record(
                "prompt_blocked", request_id, client_id=client_id, asset_id=req.asset_id,
                detail={"findings": inspection.findings, "note": inspection.sanitized_note[:200]},
            )
            req = req.model_copy(update={"operator_note": ""})

        if self.cache is None:
            return await self._run_backends(req, request_id, client_id, note_block, t0, deadline, None)

        key = cache_key(req)
        hit = await self.cache.get(key)
        if hit is not None:
            return await self._serve_cached(InferenceResponse.model_validate_json(hit), req,
                                            request_id, client_id, note_block, elapsed_ms())

        # Single-flight: si otra solicitud con la misma clave ya está en el motor
        # (p. ej. un PLC que reconecta y vacía su buffer), se espera ese resultado
        # en vez de lanzar N inferencias idénticas contra el mismo GPU.
        leader = self._inflight.get(key)
        if leader is not None:
            try:
                shared = await asyncio.wait_for(asyncio.shield(leader),
                                                timeout=max(deadline - time.perf_counter(), 0.01))
            except asyncio.TimeoutError:
                shared = None
            if shared is not None and shared.tier not in {"failsafe"}:
                return await self._serve_cached(shared, req, request_id, client_id, note_block,
                                                elapsed_ms())
            return await self._run_backends(req, request_id, client_id, note_block, t0, deadline, key)

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            resp = await self._run_backends(req, request_id, client_id, note_block, t0, deadline, key)
            fut.set_result(resp)
            return resp
        except BaseException:
            fut.set_result(None)
            raise
        finally:
            self._inflight.pop(key, None)

    async def _serve_cached(self, cached: InferenceResponse, req: InferenceRequest,
                            request_id: str, client_id: str, note_block: str | None,
                            latency_ms: float) -> InferenceResponse:
        # La acción cacheada se calculó con otra lectura del mismo bucket:
        # el paso respecto del valor actual se vuelve a verificar.
        action, reason = cached.action, note_block
        try:
            check_physics(action, req.telemetry)
        except OutputRejected as exc:
            action, reason = failsafe_hold(str(exc)), str(exc)
        resp = cached.model_copy(update={
            "request_id": request_id, "asset_id": req.asset_id, "cached": True, "tier": "cache",
            "latency_ms": latency_ms, "attempts": [], "action": action, "rejected_reason": reason,
        })
        await self._audit_response(resp, client_id, "inference_cached")
        return resp

    async def _run_backends(self, req: InferenceRequest, request_id: str, client_id: str,
                            note_block: str | None, t0: float, deadline: float,
                            key: str | None) -> InferenceResponse:
        elapsed_ms = lambda: round((time.perf_counter() - t0) * 1000, 2)  # noqa: E731
        score = complexity_score(req)
        user_prompt = build_user_prompt(req, req.operator_note)
        attempts: list[str] = []
        last_error: str | None = None

        for backend in self.plan(score):
            remaining = deadline - time.perf_counter()
            if remaining <= 0.05:
                attempts.append(f"{backend.name}:skipped(deadline)")
                break
            breaker = self.breakers[backend.name]
            timeout = min(backend_timeout(backend), remaining)
            try:
                raw = await breaker.call(
                    lambda b=backend, t=timeout: asyncio.wait_for(
                        self._generate_checked(b, user_prompt, t), timeout=t
                    )
                )
            except CircuitOpenError:
                attempts.append(f"{backend.name}:circuit_open")
                self._schedule_rewarm(backend)
                continue
            except (BackendError, OutputRejected, asyncio.TimeoutError) as exc:
                attempts.append(f"{backend.name}:{type(exc).__name__}")
                last_error = f"{backend.name}: {exc or type(exc).__name__}"
                log.warning("backend %s failed: %s", backend.name, exc)
                if breaker.state.value == "open":
                    self._schedule_rewarm(backend)
                continue

            action, _ = raw
            reason = note_block
            premise = check_premise(action, req.telemetry)
            try:
                check_physics(action, req.telemetry)
                if premise and action.action == "ADJUST_SETPOINT":
                    # Mover un setpoint por un motivo que los datos desmienten no
                    # se ejecuta, aunque el valor sea físicamente válido.
                    raise OutputRejected(premise)
            except OutputRejected as exc:
                attempts.append(f"{backend.name}:physics_rejected")
                action = failsafe_hold(str(exc))
                reason = str(exc) if note_block is None else f"{note_block}; {exc}"
                resp = InferenceResponse(
                    request_id=request_id, asset_id=req.asset_id, action=action,
                    backend=backend.name, tier=backend.tier, latency_ms=elapsed_ms(),
                    rejected_reason=reason, attempts=attempts,
                )
                await self._audit_response(resp, client_id, "inference_rejected", score)
                return resp

            attempts.append(f"{backend.name}:ok")
            resp = InferenceResponse(
                request_id=request_id, asset_id=req.asset_id, action=action,
                backend=backend.name, tier=backend.tier, latency_ms=elapsed_ms(),
                degraded=len(attempts) > 1, rejected_reason=reason, attempts=attempts,
                warnings=[premise] if premise else [],
            )
            if self.cache is not None and key is not None and resp.rejected_reason is None:
                await self.cache.set(key, resp.model_dump_json())
            await self._audit_response(resp, client_id, "inference_ok", score)
            return resp

        resp = InferenceResponse(
            request_id=request_id, asset_id=req.asset_id,
            action=failsafe_hold("no valid answer: " + (", ".join(attempts) or "no edge backend for this complexity")),
            backend="none", tier="failsafe", latency_ms=elapsed_ms(), degraded=True,
            rejected_reason="; ".join(x for x in (note_block, last_error) if x) or None,
            attempts=attempts,
        )
        await self._audit_response(resp, client_id, "inference_failsafe", score)
        return resp

    async def _generate_checked(self, backend: InferenceBackend, user_prompt: str, timeout: float):
        # El parseo va dentro de la llamada protegida por el breaker: un backend
        # que responde basura con HTTP 200 está tan caído como uno que no responde.
        raw = await backend.generate(SYSTEM_PROMPT, user_prompt, _SCHEMA, timeout)
        return parse_action(raw), raw

    async def _audit_response(self, resp: InferenceResponse, client_id: str, event: str,
                              score: float | None = None) -> None:
        a = resp.action
        await self.audit.record(
            event, resp.request_id, client_id=client_id, asset_id=resp.asset_id,
            backend=resp.backend, tier=resp.tier, action=a.action, target_tag=a.target_tag,
            new_value=a.new_value, latency_ms=resp.latency_ms,
            rejected_reason=resp.rejected_reason,
            detail={"attempts": resp.attempts, "complexity": score,
                    "confidence": a.confidence, "rationale": a.rationale},
        )

    def _schedule_rewarm(self, backend: InferenceBackend) -> None:
        """Con el breaker abierto, recarga el modelo en segundo plano.

        Un motor que se reinicia vuelve sin el modelo en VRAM (~35 s de carga en
        frío medidos). Las sondas half-open tienen el timeout corto del lazo de
        control y nunca alcanzarían a cargarlo; esta tarea sí, con timeout largo,
        y cuando termina la siguiente sonda encuentra el modelo listo.
        """
        if backend.tier != "edge" or not hasattr(backend, "warmup"):
            return
        task = self._rewarm.get(backend.name)
        if task is not None and not task.done():
            return
        # Ya recargado hace poco: el breaker sigue abierto solo porque la sonda
        # half-open aún no llega. Recargar otra vez no adelanta nada.
        if time.monotonic() - self._rewarm_ok_at.get(backend.name, -1e9) < self.settings.breaker_reset_timeout_s:
            return

        async def _run():
            try:
                await backend.warmup()
                self._rewarm_ok_at[backend.name] = time.monotonic()
                log.info("re-warmed %s", backend.name)
            except Exception as exc:
                log.warning("re-warm of %s failed: %s", backend.name, exc)
                await asyncio.sleep(5)  # el motor sigue caído; reintenta en la próxima falla

        self._rewarm[backend.name] = asyncio.create_task(_run())

    async def warmup(self) -> dict[str, str]:
        """Carga los modelos del borde sin pasar por el breaker ni el deadline."""
        status = {}
        for b in self.backends:
            if b.tier != "edge" or not hasattr(b, "warmup"):
                continue
            try:
                await b.warmup()
                status[b.name] = "ready"
            except Exception as exc:  # el gateway arranca igual; el breaker se encarga después
                status[b.name] = f"warmup failed: {exc}"
                log.warning("warmup of %s failed: %s", b.name, exc)
        return status

    def breaker_status(self) -> list[dict]:
        return [b.snapshot() for b in self.breakers.values()]

    async def aclose(self) -> None:
        for t in self._rewarm.values():
            t.cancel()
        for b in self.backends:
            await b.aclose()


def backend_timeout(backend: InferenceBackend) -> float:
    settings = getattr(backend, "settings", None)
    return settings.timeout_s if settings is not None else 2.0
