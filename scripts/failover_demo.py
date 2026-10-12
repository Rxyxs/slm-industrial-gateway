"""Caída y recuperación del motor de inferencia con el stack de deploy/edge levantado.

Envía una solicitud por segundo al gateway desplegado (hora real de la planta
de flotación), detiene el contenedor de Ollama a mitad de la corrida y lo vuelve
a levantar. Registra por solicitud: tier, latencia, acción y estado del circuit
breaker, para ver en datos lo que el diseño promete:
- el gateway nunca devuelve error al SCADA ni excede el deadline,
- el breaker se abre tras N fallas y deja de esperar al motor caído,
- vuelve a cerrar solo, con una solicitud de prueba, cuando el motor regresa.

    python scripts/failover_demo.py --csv <ruta>/MiningProcess_Flotation_Plant_Database.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gateway.tags import TAG_REGISTRY  # noqa: E402


def docker(*args: str) -> None:
    subprocess.run(["docker", *args], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="local-test-key")
    ap.add_argument("--container", default="slm-edge-ollama-1")
    ap.add_argument("--stop-at", type=float, default=10.0)
    ap.add_argument("--start-at", type=float, default=35.0)
    ap.add_argument("--duration", type=float, default=80.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/reports/failover_timeline.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, decimal=",", parse_dates=["date"])
    hourly = df.groupby("date").mean(numeric_only=True)
    headers = {"X-API-Key": args.key}

    events: list[dict] = []
    t0 = time.perf_counter()

    def chaos():
        time.sleep(args.stop_at)
        docker("stop", args.container)
        events.append({"t": round(time.perf_counter() - t0, 2), "event": "engine stopped"})
        time.sleep(args.start_at - args.stop_at)
        docker("start", args.container)
        events.append({"t": round(time.perf_counter() - t0, 2), "event": "engine started"})

    threading.Thread(target=chaos, daemon=True).start()
    rows = []
    with httpx.Client(base_url=args.url, headers=headers, timeout=10) as client:
        i = 0
        while (elapsed := time.perf_counter() - t0) < args.duration:
            row = hourly.iloc[i % len(hourly)]
            body = {"asset_id": "flot-col-01", "telemetry": [
                {"tag": s.tag, "value": round(float(row[s.source_column]), 4)}
                for s in TAG_REGISTRY.values()]}
            t_req = time.perf_counter()
            try:
                r = client.post("/v1/infer", json=body)
                status, data = r.status_code, r.json()
            except httpx.HTTPError as exc:
                status, data = 0, {"error": str(exc)}
            wall_ms = (time.perf_counter() - t_req) * 1000
            breaker = client.get("/v1/backends").json()["backends"][0]["state"]
            rows.append({"t": round(elapsed, 2), "http": status, "tier": data.get("tier"),
                         "action": (data.get("action") or {}).get("action"),
                         "wall_ms": round(wall_ms, 1), "breaker": breaker,
                         "attempts": data.get("attempts")})
            print(rows[-1])
            i += 1
            time.sleep(max(0.0, 1.0 - (time.perf_counter() - t_req)))

    summary = {
        "requests": len(rows),
        "non_200": sum(r["http"] != 200 for r in rows),
        "max_wall_ms": max(r["wall_ms"] for r in rows),
        "tiers": {t: sum(r["tier"] == t for r in rows) for t in {r["tier"] for r in rows}},
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "timeline": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
