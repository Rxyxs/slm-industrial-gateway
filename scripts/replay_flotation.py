"""Reproduce horas reales de la planta de flotación por MQTT contra el gateway desplegado.

Publica en plant/<asset>/telemetry y espera cada respuesta en
plant/<asset>/recommendation, midiendo el tiempo de ida y vuelta completo
(broker -> gateway -> SLM -> validación -> broker).

    python scripts/replay_flotation.py --csv <ruta>/MiningProcess_Flotation_Plant_Database.csv --n 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiomqtt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gateway.tags import TAG_REGISTRY  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--asset", default="flot-col-01")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, decimal=",", parse_dates=["date"])
    hourly = df.groupby("date").mean(numeric_only=True).head(args.n)

    rtts, actions = [], []
    async with aiomqtt.Client(args.host, args.port, identifier="flotation-replay") as client:
        await client.subscribe(f"plant/{args.asset}/#", qos=1)
        for ts, row in hourly.iterrows():
            payload = {
                "asset_id": args.asset, "operator_note": args.note,
                "telemetry": [{"tag": s.tag, "value": round(float(row[s.source_column]), 4),
                               "ts": ts.isoformat()}
                              for s in TAG_REGISTRY.values() if s.source_column in row.index],
            }
            t0 = time.perf_counter()
            await client.publish(f"plant/{args.asset}/telemetry", json.dumps(payload), qos=1)
            async for msg in client.messages:
                topic = str(msg.topic)
                if topic.endswith("/recommendation") or topic.endswith("/rejected"):
                    break
            rtt = (time.perf_counter() - t0) * 1000
            out = json.loads(msg.payload)
            rtts.append(rtt)
            a = out.get("action", {})
            actions.append(a.get("action", "REJECTED"))
            print(f"{ts}  silica={row['% Silica Concentrate']:.2f}%  -> {a.get('action')} "
                  f"{a.get('target_tag') or ''} {a.get('new_value') or ''}  tier={out.get('tier')} "
                  f"rtt={rtt:.0f}ms  note={out.get('rejected_reason')}")

    print(json.dumps({"n": len(rtts), "rtt_ms_p50": round(statistics.median(rtts), 1),
                      "rtt_ms_max": round(max(rtts), 1),
                      "actions": {a: actions.count(a) for a in set(actions)}}))


if __name__ == "__main__":
    if sys.platform == "win32":
        # aiomqtt necesita add_reader/add_writer, que el loop Proactor de Windows no tiene.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
