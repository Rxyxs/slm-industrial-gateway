"""Ablación de utilidad sobre 120 horas reales (40 ADJUST / 40 HOLD / 40 HOLD_NEAR_MISS) con SLMs reales en Ollama.

Variantes:
  A snapshot         solo la telemetría de la hora (el gateway original)
  B context          hechos (laboratorio, soft sensor, tendencias) + procedimiento
  C context_fewshot  B + 5 ejemplos resueltos
  D tools            mismo procedimiento, pero el modelo pide los hechos con function calling
  E context_derived  B + el paso 1 del procedimiento (estado de la sílice) calculado en código
  F context_gated    E + la gramática solo ofrece ADJUST si el procedimiento (en código) lo permite

A-C pasan por el mismo `InferenceRouter` de producción (deadline, física, premisas, citas).
D usa `tool_agent.decide_with_tools` y después las mismas verificaciones, sin deadline
(se reporta su latencia).

    python scripts/eval_utility.py --models qwen2.5:1.5b-instruct qwen2.5:3b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gateway.audit import AuditLog  # noqa: E402
from src.gateway.backends import OllamaBackend  # noqa: E402
from src.gateway.config import BackendSettings, GatewaySettings  # noqa: E402
from src.gateway.router import InferenceRouter  # noqa: E402
from src.gateway.schemas import setpoint_bounds  # noqa: E402
from src.gateway.soft_sensor import SoftSensor  # noqa: E402
from src.gateway.tool_agent import decide_with_tools  # noqa: E402
from src.gateway.utility_eval import load_cases, outcome_from, summarize  # noqa: E402
from src.gateway.validation import (  # noqa: E402
    OutputRejected, check_citations, check_physics, check_premise, failsafe_hold, parse_action,
)

MODES = {"A": "snapshot", "B": "context", "C": "context_fewshot", "D": "tools", "E": "context_derived",
         "F": "context_gated"}


async def run_router(model: str, url: str, mode: str, cases, deadline: float):
    backend = OllamaBackend(BackendSettings(name=model, kind="ollama", base_url=url, model=model,
                                            tier="edge", timeout_s=deadline))
    await backend.warmup()
    settings = GatewaySettings(deadline_s=deadline, breaker_failure_threshold=10_000,
                               audit_db_path=":memory:", prompt_mode=mode)
    audit = AuditLog(":memory:")
    router = InferenceRouter(settings, [backend], audit, cache=None)
    out = []
    for case in cases:
        resp = await router.infer(case.request)
        proposed = (resp.action.action == "ADJUST_SETPOINT"
                    or any(a.endswith("physics_rejected") for a in resp.attempts))
        out.append(outcome_from(case, resp.action, proposed_adjust=proposed, rejected=resp.rejected_reason,
                                warnings=resp.warnings, tier=resp.tier, latency_ms=resp.latency_ms))
    ok, _ = audit.verify_chain()
    audit.close()
    await backend.aclose()
    return out, ok


async def run_tools(model: str, url: str, cases, sensor):
    out = []
    async with httpx.AsyncClient(base_url=url) as client:
        for case in cases:
            ctx = case.context(sensor)
            try:
                raw, trace = await decide_with_tools(client, model, case.request.asset_id, ctx,
                                                     setpoint_bounds(case.request.telemetry))
            except httpx.HTTPError as exc:
                out.append(outcome_from(case, failsafe_hold(str(exc)), proposed_adjust=False,
                                        rejected=str(exc), warnings=[], tier="failsafe", latency_ms=0.0))
                continue
            try:
                action = parse_action(raw)
            except OutputRejected as exc:
                out.append(outcome_from(case, failsafe_hold(str(exc)), proposed_adjust=False, rejected=str(exc),
                                        warnings=[], tier="failsafe", latency_ms=trace["latency_ms"],
                                        tools=trace["tools"]))
                continue
            proposed = action.action == "ADJUST_SETPOINT"
            premise = check_premise(action, case.request.telemetry)
            cite_reject, cite_warn = check_citations(action, ctx)
            rejected = None
            try:
                check_physics(action, case.request.telemetry)
                if premise and proposed:
                    raise OutputRejected(premise)
                if cite_reject:
                    raise OutputRejected(cite_reject)
            except OutputRejected as exc:
                rejected, action = str(exc), failsafe_hold(str(exc))
            out.append(outcome_from(case, action, proposed_adjust=proposed, rejected=rejected,
                                    warnings=[w for w in (premise, cite_warn) if w], tier="edge",
                                    latency_ms=trace["latency_ms"], tools=trace["tools"]))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"], choices=list(MODES))
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--deadline", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/reports/utility_eval.json"))
    args = ap.parse_args()

    cases = load_cases(limit=args.limit)
    sensor = SoftSensor.load()
    results = json.loads(args.out.read_text())["results"] if args.out.exists() else []
    done = {(r["model"], r["variant"]) for r in results}

    for model in args.models:
        for v in args.variants:
            if (model, v) in done:
                continue
            t0 = time.perf_counter()
            if MODES[v] == "tools":
                outcomes, chain_ok = await run_tools(model, args.url, cases, sensor), None
            else:
                outcomes, chain_ok = await run_router(model, args.url, MODES[v], cases, args.deadline)
            summary = summarize(outcomes)
            rec = {"model": model, "variant": v, "mode": MODES[v], "deadline_s": args.deadline,
                   **summary, "audit_chain_ok": chain_ok, "wall_s": round(time.perf_counter() - t0, 1),
                   "rejections": sorted({(o.rejected or "")[:90] for o in outcomes if o.rejected}),
                   "warnings": sorted({w[:90] for o in outcomes for w in o.warnings}),
                   "tools_used": sorted({t for o in outcomes for t in o.tools}),
                   "cases": [o.record() for o in outcomes]}
            results.append(rec)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"cases": "data/evals/utility_cases.json", "results": results},
                                           indent=1, default=str))
            print(json.dumps({k: rec[k] for k in ("model", "variant", "adjust_recall_pct",
                                                  "adjust_proposed_pct", "adjust_wrong_direction_pct", "false_adjust_pct", "near_miss_false_adjust_pct",
                                                  "alarm_on_calm_pct", "failsafe_pct",
                                                  "citation_or_premise_warning_pct", "latency_ms_p50",
                                                  "latency_ms_p95")}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
