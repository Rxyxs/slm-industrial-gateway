"""Benchmark del gateway contra SLMs reales servidos por Ollama, con telemetría real.

Fuente: dataset público de una planta de flotación de hierro (Kaggle, "Quality
Prediction in a Mining Process"). Se promedia cada hora (la sílice de
laboratorio es horaria) y se muestrean N horas con semilla fija. Cada hora se
convierte en un InferenceRequest con los 10 tags del registro y pasa por el
mismo `InferenceRouter` que usa el gateway en producción (deadline, breaker,
validación de schema y física, auditoría).

No hay etiqueta de "acción correcta" en el dataset, así que no se reporta
exactitud. Se mide lo que el gateway garantiza o no: cuántas salidas cumplen el
contrato, cuántas se rechazan por física, cuántas caen en failsafe y la latencia.

    python scripts/bench_gateway.py --csv <ruta>/MiningProcess_Flotation_Plant_Database.csv \
        --models qwen2.5:0.5b-instruct qwen2.5:1.5b-instruct --n 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gateway.audit import AuditLog  # noqa: E402
from src.gateway.backends import BackendError, OllamaBackend  # noqa: E402
from src.gateway.breaker import CircuitBreaker  # noqa: E402
from src.gateway.config import BackendSettings, GatewaySettings  # noqa: E402
from src.gateway.router import InferenceRouter  # noqa: E402
from src.gateway.schemas import InferenceRequest  # noqa: E402
from src.gateway.security import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from src.gateway.schemas import model_output_json_schema  # noqa: E402
from src.gateway.tags import TAG_REGISTRY  # noqa: E402
from src.gateway.validation import OutputRejected, check_physics, check_premise, parse_action  # noqa: E402


def load_hours(csv: Path, n: int, seed: int) -> dict[str, pd.DataFrame]:
    """Dos conjuntos: horas al azar, y horas del decil superior de sílice en
    concentrado (donde un operador sí movería reactivos)."""
    df = pd.read_csv(csv, decimal=",", parse_dates=["date"])
    hourly = df.groupby("date").mean(numeric_only=True)
    cut = hourly["% Silica Concentrate"].quantile(0.9)
    high = hourly[hourly["% Silica Concentrate"] >= cut]
    return {
        "random": hourly.sample(n=min(n, len(hourly)), random_state=seed).sort_index(),
        "high_silica": high.sample(n=min(n, len(high)), random_state=seed).sort_index(),
    }


def to_request(ts, row) -> InferenceRequest:
    telemetry = [{"tag": spec.tag, "value": float(row[spec.source_column])}
                 for spec in TAG_REGISTRY.values() if spec.source_column in row.index]
    return InferenceRequest(asset_id="flot-col-01", telemetry=telemetry)


async def classify_raw(backend: OllamaBackend, req: InferenceRequest) -> str:
    """Llamada directa (sin router) para clasificar POR QUÉ falla una salida."""
    try:
        raw = await backend.generate(SYSTEM_PROMPT, build_user_prompt(req, ""),
                                     model_output_json_schema(), timeout_s=30)
    except BackendError as exc:
        return "engine: " + ("timeout >30s" if "timeout" in str(exc) else "error")
    try:
        action = parse_action(raw)
    except OutputRejected as exc:
        return "shape: " + str(exc).split(":")[0].replace("schema violation at ", "")
    try:
        check_physics(action, req.telemetry)
    except OutputRejected as exc:
        msg = str(exc)
        return "physics: " + ("step" if "max_step" in msg else "range" if "outside" in msg else "no reading")
    if check_premise(action, req.telemetry):
        return "valid, premise contradicted"
    return "valid"


async def bench_model(model: str, url: str, requests: list[InferenceRequest], deadline: float) -> dict:
    bs = BackendSettings(name=model, kind="ollama", base_url=url, model=model, tier="edge",
                         timeout_s=deadline)
    backend = OllamaBackend(bs)
    await backend.warmup()
    settings = GatewaySettings(deadline_s=deadline, breaker_failure_threshold=10_000,
                               audit_db_path=":memory:")
    audit = AuditLog(":memory:")
    router = InferenceRouter(settings, [backend], audit, cache=None)

    latencies, tiers, actions, outcomes = [], Counter(), Counter(), Counter()
    premise_warnings = 0
    for req in requests:
        resp = await router.infer(req)
        latencies.append(resp.latency_ms)
        tiers[resp.tier] += 1
        actions[resp.action.action] += 1
        premise_warnings += bool(resp.warnings)
        if resp.tier == "failsafe":
            outcomes[resp.attempts[-1].split(":")[1] if resp.attempts else "deadline"] += 1
        elif resp.rejected_reason:
            outcomes["physics_rejected"] += 1
        else:
            outcomes["valid"] += 1

    # Segunda pasada sin deadline para desglosar los motivos de rechazo.
    reasons = Counter()
    for req in requests:
        reasons[await classify_raw(backend, req)] += 1

    chain_ok, _ = audit.verify_chain()
    audit.close()
    await backend.aclose()
    n = len(requests)
    q = statistics.quantiles(latencies, n=20, method="inclusive")
    return {
        "model": model, "n": n, "deadline_s": deadline,
        "valid_pct": round(100 * outcomes["valid"] / n, 1),
        "physics_rejected_pct": round(100 * outcomes["physics_rejected"] / n, 1),
        "contract_failure_pct": round(100 * sum(v for k, v in outcomes.items()
                                                if k not in {"valid", "physics_rejected"}) / n, 1),
        "premise_contradicted_pct": round(100 * premise_warnings / n, 1),
        "outcomes": dict(outcomes), "rejection_breakdown_no_deadline": dict(reasons),
        "actions": dict(actions), "tiers": dict(tiers),
        "latency_ms": {"p50": round(statistics.median(latencies), 1), "p95": round(q[18], 1),
                       "max": round(max(latencies), 1)},
        "audit_chain_ok": chain_ok,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--deadline", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/reports/gateway_bench.json"))
    args = ap.parse_args()

    sets = load_hours(args.csv, args.n, args.seed)
    # Reanudable: cada (conjunto, modelo) terminado se guarda de inmediato.
    results = json.loads(args.out.read_text())["results"] if args.out.exists() else []
    done = {(r["set"], r["model"]) for r in results}

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"source": "Kaggle: Quality Prediction in a Mining Process",
                                        "seed": args.seed, "results": results}, indent=2))

    for set_name, hours in sets.items():
        requests = [to_request(ts, row) for ts, row in hours.iterrows()]
        print(f"[{set_name}] {len(requests)} hourly snapshots, "
              f"silica conc. mean {hours['% Silica Concentrate'].mean():.2f}%")
        for model in args.models:
            if (set_name, model) in done:
                continue
            t0 = time.perf_counter()
            r = await bench_model(model, args.url, requests, args.deadline)
            r["set"] = set_name
            r["wall_s"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            save()
            print(json.dumps({k: r[k] for k in ("set", "model", "valid_pct", "physics_rejected_pct",
                                                "contract_failure_pct", "premise_contradicted_pct", "actions", "latency_ms",
                                                "rejection_breakdown_no_deadline")}))

    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
