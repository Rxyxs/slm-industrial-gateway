"""API REST del gateway de planta.

POST /v1/infer          inferencia síncrona acotada por deadline (lazo de control)
POST /v1/jobs           encola una inferencia diferida -> 202 + job_id
GET  /v1/jobs/{job_id}  estado / resultado de un job
POST /v1/lab            resultado de laboratorio de sílice -> contexto del activo
GET  /v1/context/{asset_id}  hechos que vería el modelo ahora para ese activo
GET  /v1/backends       estado de cada circuit breaker
GET  /v1/audit          últimas N decisiones + verificación de la cadena de hashes
GET  /v1/schema         JSON Schema de ControlAction (SCADA) y de ModelOutput (guided decoding)
GET  /health            liveness (no toca backends)
GET  /metrics           Prometheus

Todo endpoint salvo /health y /metrics exige `X-API-Key` cuando hay claves
configuradas. mTLS se aplica antes, en el transporte (`serve.py`).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from src.gateway.audit import AuditLog
from src.gateway.context import ContextStore, LabResult, build_context
from src.gateway.backends import build_backend
from src.gateway.config import GatewaySettings, load_settings
from src.gateway.jobs import InMemoryJobQueue, RedisJobQueue
from src.gateway.protocols.field import FieldWriter
from src.gateway.resilience import RedisCache, TokenBucketLimiter, TTLCache
from src.gateway.router import InferenceRouter
from src.gateway.schemas import (
    InferenceRequest, InferenceResponse, JobAccepted, JobStatus, LabIn, control_action_json_schema,
    model_output_json_schema,
)
from src.gateway.security import ApiKeyAuth


def create_app(settings: GatewaySettings | None = None, backends=None, redis=None,
               field_writer: FieldWriter | None = None) -> FastAPI:
    settings = settings or load_settings()
    registry = CollectorRegistry()
    m_requests = Counter("slmgw_inference_total", "Inferences by tier and outcome",
                         ["tier", "action", "degraded"], registry=registry)
    m_latency = Histogram("slmgw_inference_latency_seconds", "End-to-end inference latency",
                          ["tier"], registry=registry,
                          buckets=(0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10))
    m_limited = Counter("slmgw_rate_limited_total", "Requests rejected by rate limiter",
                        registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal redis
        if redis is None and settings.redis_url:
            import redis.asyncio as aioredis
            redis = aioredis.from_url(settings.redis_url)
        audit = AuditLog(settings.audit_db_path)
        cache = (RedisCache(redis, settings.cache_ttl_s) if redis is not None
                 else TTLCache(settings.cache_ttl_s, settings.cache_max_items))
        context_store = ContextStore(settings.context_window_h)
        router = InferenceRouter(
            settings, backends if backends is not None else [build_backend(b) for b in settings.backends],
            audit, cache, context_store=context_store,
        )
        queue = (RedisJobQueue(redis, router.infer) if redis is not None
                 else InMemoryJobQueue(router.infer))
        await queue.start()
        # La carga del modelo corre en segundo plano: /health responde de
        # inmediato y las inferencias previas a la carga caen en failsafe.
        app.state.warmup = {}
        warm_task = asyncio.create_task(router.warmup())
        warm_task.add_done_callback(
            lambda t: app.state.warmup.update(t.result() if not t.cancelled() and not t.exception() else {}))
        app.state.settings = settings
        app.state.audit = audit
        app.state.router = router
        app.state.queue = queue
        app.state.auth = ApiKeyAuth(settings.api_keys)
        app.state.limiter = TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
        app.state.writer = field_writer or FieldWriter(None, settings.allow_plc_writeback)
        yield
        warm_task.cancel()
        await queue.stop()
        await router.aclose()
        audit.close()

    app = FastAPI(title="Industrial SLM Gateway", version="0.2.0", lifespan=lifespan)

    async def client_id(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        cid = request.app.state.auth.verify(x_api_key)
        if cid is None:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
        ok, wait = request.app.state.limiter.allow(cid)
        if not ok:
            m_limited.inc()
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": str(max(1, round(wait)))})
        return cid

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail, "type": "HTTPException"},
                            status_code=exc.status_code, headers=exc.headers)

    @app.post("/v1/infer", response_model=InferenceResponse)
    async def infer(req: InferenceRequest, request: Request, cid: str = Depends(client_id)):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        t0 = time.perf_counter()
        resp = await request.app.state.router.infer(req, client_id=cid, request_id=rid)
        written, why = await request.app.state.writer.apply(req, resp)
        m_latency.labels(resp.tier).observe(time.perf_counter() - t0)
        m_requests.labels(resp.tier, resp.action.action, str(resp.degraded).lower()).inc()
        return JSONResponse(resp.model_dump(mode="json"),
                            headers={"X-Request-ID": rid, "X-PLC-Written": str(written).lower(),
                                     "X-PLC-Write-Note": why})

    @app.post("/v1/lab", status_code=202)
    async def post_lab(lab: LabIn, request: Request, _: str = Depends(client_id)):
        if lab.asset_id is None:
            raise HTTPException(status_code=422, detail="asset_id is required")
        store = request.app.state.router.context_store
        store.add_lab(lab.asset_id, LabResult(lab.sampled_at, lab.silica_pct, lab.delay_h))
        return {"accepted": True, "labs_held": len(store.labs(lab.asset_id))}

    @app.get("/v1/context/{asset_id}")
    async def get_context(asset_id: str, request: Request, _: str = Depends(client_id)):
        router = request.app.state.router
        store = router.context_store
        window, labs = store.window(asset_id), store.labs(asset_id)
        ctx = build_context(window, labs, router.soft_sensor) if window else None
        return {"asset_id": asset_id, "hours": len(window), "labs": len(labs),
                "prompt_mode": request.app.state.settings.prompt_mode,
                "facts": [f.__dict__ for f in ctx.facts] if ctx else []}

    @app.post("/v1/jobs", response_model=JobAccepted, status_code=202)
    async def submit_job(req: InferenceRequest, request: Request, cid: str = Depends(client_id)):
        job_id = await request.app.state.queue.submit(req, cid)
        return JobAccepted(job_id=job_id)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    async def job_status(job_id: str, request: Request, _: str = Depends(client_id)):
        status = await request.app.state.queue.get(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="job not found")
        return status

    @app.get("/v1/backends")
    async def backends_status(request: Request, _: str = Depends(client_id)):
        return {"backends": request.app.state.router.breaker_status(),
                "warmup": request.app.state.warmup,
                "allow_cloud": request.app.state.settings.allow_cloud}

    @app.get("/v1/audit")
    async def audit(request: Request, limit: int = 50, _: str = Depends(client_id)):
        log = request.app.state.audit
        ok, broken_at = log.verify_chain()
        rows = log.recent(min(max(limit, 1), 500))
        return {"chain_ok": ok, "broken_at_seq": broken_at,
                "entries": [{k: (v.isoformat() if hasattr(v, "isoformat") else v)
                             for k, v in r.items()} for r in rows]}

    @app.get("/v1/schema")
    async def schema():
        return {"scada_action": control_action_json_schema(),
                "model_output": model_output_json_schema()}

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": app.version}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
