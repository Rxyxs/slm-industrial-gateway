"""Variante experimental: el modelo pide los datos con herramientas en vez de recibirlos inyectados.

Es la fase 4 de la estrategia de contexto: ¿le conviene a un SLM pequeño consultar la
tendencia con function calling (`get_lab_silica()`, `get_setpoint("amina_flow")`, ...) en
vez de recibir todos los hechos de golpe? Para que la comparación sea justa, las
herramientas devuelven exactamente los mismos hechos (con los mismos ids) que el modo de
inyección, y la decisión final pasa por el mismo schema y las mismas verificaciones.
Lo único que cambia es quién decide qué datos mirar.

Flujo (Ollama `/api/chat` con `tools`):
1. hasta `max_rounds` turnos con herramientas y sin `format`;
2. un turno final con el JSON Schema de `ModelOutput` (evidence = ids del contexto).

No está conectado al router de producción: cuesta 2 o más viajes al motor por decisión,
y se evalúa con `scripts/eval_utility.py` antes de decidir si vale la pena.
"""

from __future__ import annotations

import json
import time

import httpx

from src.gateway.context import PlantContext
from src.gateway.prompts import CONTEXT_SYSTEM
from src.gateway.schemas import model_output_json_schema

TOOLS = [
    {"type": "function", "function": {
        "name": "get_lab_silica",
        "description": "Latest and previous lab results of silica in concentrate, and how many "
                       "consecutive results are high.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_soft_sensor",
        "description": "Soft-sensor estimate of silica in the concentrate right now, with its typical error.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_setpoint",
        "description": "Current value, recent mean, trend, allowed range, room to raise and max "
                       "change per step of one writable setpoint.",
        "parameters": {"type": "object", "properties": {
            "tag": {"type": "string",
                    "enum": ["amina_flow", "starch_flow", "column_01_air_flow", "column_01_level"]}},
            "required": ["tag"]}}},
    {"type": "function", "function": {
        "name": "get_pulp_conditions",
        "description": "Pulp pH and density and whether each is inside its normal range.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

TOOL_SYSTEM = CONTEXT_SYSTEM.replace(
    "You receive numbered FACTS computed from plant data. Use only those facts.",
    "You do not see the plant data directly: call the tools to read it. Each tool returns "
    "numbered FACTS. Read the silica lab results first, then whatever else the procedure "
    "needs. Use only facts returned by the tools.")


def run_tool(ctx: PlantContext, name: str, args: dict) -> str:
    if name == "get_lab_silica":
        facts = [f for f in ctx.facts if f.kind in {"lab_last", "lab_prev", "lab_streak"}]
    elif name == "get_soft_sensor":
        facts = [f for f in ctx.facts if f.kind == "soft_sensor"]
    elif name == "get_setpoint":
        facts = [f for f in ctx.facts if f.kind == "setpoint" and f.tag == args.get("tag")]
    elif name == "get_pulp_conditions":
        facts = [f for f in ctx.facts if f.kind == "range"]
    else:
        return f"unknown tool {name}"
    return "\n".join(f"[{f.id}] {f.text}" for f in facts) or "no data"


async def decide_with_tools(client: httpx.AsyncClient, model: str, asset_id: str, ctx: PlantContext,
                            bounds: dict | None = None, timeout_s: float = 30.0,
                            max_rounds: int = 4) -> tuple[str, dict]:
    """Devuelve (JSON crudo de la decisión, traza con herramientas usadas y latencia)."""
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": TOOL_SYSTEM},
                {"role": "user", "content": f"asset_id: {asset_id}. Decide the control action for this hour."}]
    called: list[str] = []
    opts = {"temperature": 0, "seed": 7, "num_predict": 160}
    for _ in range(max_rounds):
        r = await client.post("/api/chat", json={"model": model, "messages": messages, "tools": TOOLS,
                                                 "stream": False, "options": opts}, timeout=timeout_s)
        r.raise_for_status()
        msg = r.json()["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
        for c in calls:
            fn = c["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            called.append(fn["name"] + (f"({args.get('tag')})" if args.get("tag") else ""))
            messages.append({"role": "tool", "content": run_tool(ctx, fn["name"], args)})
    messages.append({"role": "user", "content": "Apply the operating procedure and return the JSON."})
    r = await client.post("/api/chat", json={"model": model, "messages": messages, "stream": False,
                                             "format": model_output_json_schema(ctx.ids, bounds), "options": opts},
                          timeout=timeout_s)
    r.raise_for_status()
    return r.json()["message"]["content"], {"tools": called,
                                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
