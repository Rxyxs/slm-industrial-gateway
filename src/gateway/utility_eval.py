"""Evaluación de utilidad: ¿el gateway propone el ajuste que la regla operativa pide, y solo entonces?

Las pruebas de seguridad (tests/gateway) verifican que el gateway nunca ejecuta algo
inválido. Esto mide lo contrario: si, con el contexto inyectado, el SLM se atreve a
proponer el cambio correcto. Se usa desde `scripts/eval_utility.py` (benchmark completo)
y desde `tests/evals/test_utility.py` (umbral mínimo contra un Ollama real).

Cada caso es una hora real fuera de muestra con su etiqueta según `policy.label_case`:
- ADJUST: acierto = la acción que el gateway *deja pasar* es ADJUST_SETPOINT sobre
  amina_flow hacia arriba. Un ajuste propuesto pero rechazado por física o por citas no
  cuenta como acierto (en planta no se ejecutaría).
- HOLD: acierto = no se ajusta nada. HOLD o RAISE_ALARM cuentan como no ajustar, y las
  alarmas en horas tranquilas se reportan aparte (son ruido para el operador).
- HOLD_NEAR_MISS: sílice alta pero el procedimiento dice esperar (una sola muestra alta, o
  ya bajando). Acierto = no ajustar. Es la prueba de que el modelo sigue el procedimiento
  y no solo reacciona a un número alto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.gateway.context import HourRecord, LabResult, PlantContext, build_context
from src.gateway.schemas import ControlAction, InferenceRequest
from src.gateway.soft_sensor import SoftSensor

CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "evals" / "utility_cases.json"


@dataclass
class Case:
    ts: str
    expected: str
    request: InferenceRequest
    amina_now: float

    def context(self, sensor: SoftSensor | None) -> PlantContext:
        window = [HourRecord(h.ts, h.values) for h in self.request.history]
        labs = [LabResult(l.sampled_at, l.silica_pct, l.delay_h) for l in self.request.lab_results]
        return build_context(window, labs, sensor)


def load_cases(path: Path = CASES_PATH, limit: int | None = None) -> list[Case]:
    data = json.loads(Path(path).read_text())
    out = []
    for c in data["cases"]:
        ts = datetime.fromisoformat(c["ts"])
        req = InferenceRequest(
            asset_id="flot-col-01",
            telemetry=[{**t, "ts": ts} for t in c["telemetry"]],
            history=c["history"], lab_results=c["lab_results"],
        )
        amina = next(t["value"] for t in c["telemetry"] if t["tag"] == "amina_flow")
        out.append(Case(c["ts"], c["expected"], req, amina))
    if limit:
        per = max(1, limit // 3)
        out = [c for label in ("ADJUST", "HOLD", "HOLD_NEAR_MISS")
               for c in [x for x in out if x.expected == label][:per]]
    return out


@dataclass
class Outcome:
    expected: str
    final: str              # acción que el gateway deja pasar
    target: str | None
    direction: int          # +1 sube, -1 baja, 0 sin ajuste
    proposed_adjust: bool   # el modelo propuso un ajuste (aunque se haya rechazado)
    rejected: str | None
    warnings: list[str]
    tier: str
    latency_ms: float
    tools: list[str] = field(default_factory=list)
    ts: str = ""
    evidence: list[str] = field(default_factory=list)
    new_value: float | None = None
    reason_code: str = ""

    def record(self) -> dict:
        return {"ts": self.ts, "expected": self.expected, "final": self.final, "target": self.target,
                "new_value": self.new_value, "reason_code": self.reason_code, "evidence": self.evidence,
                "proposed_adjust": self.proposed_adjust, "correct": self.correct,
                "rejected": self.rejected, "tier": self.tier, "latency_ms": self.latency_ms,
                "tools": self.tools}

    @property
    def correct(self) -> bool:
        if self.expected == "ADJUST":
            return self.final == "ADJUST_SETPOINT" and self.target == "amina_flow" and self.direction > 0
        return self.final != "ADJUST_SETPOINT"


def outcome_from(case: Case, action: ControlAction, *, proposed_adjust: bool, rejected: str | None,
                 warnings: list[str], tier: str, latency_ms: float, tools=None) -> Outcome:
    direction = 0
    if action.action == "ADJUST_SETPOINT" and action.target_tag == "amina_flow":
        direction = 1 if action.new_value > case.amina_now else -1
    elif action.action == "ADJUST_SETPOINT":
        direction = 1
    return Outcome(case.expected, action.action, action.target_tag, direction, proposed_adjust,
                   rejected, warnings, tier, latency_ms, list(tools or []), case.ts,
                   list(action.evidence), action.new_value, action.reason_code)


def summarize(outcomes: list[Outcome]) -> dict:
    adj = [o for o in outcomes if o.expected == "ADJUST"]
    hold = [o for o in outcomes if o.expected == "HOLD"]
    near = [o for o in outcomes if o.expected == "HOLD_NEAR_MISS"]
    pct = lambda n, d: round(100 * n / d, 1) if d else None  # noqa: E731
    lat = sorted(o.latency_ms for o in outcomes)
    return {
        "n_adjust_cases": len(adj), "n_hold_cases": len(hold), "n_near_miss_cases": len(near),
        "adjust_recall_pct": pct(sum(o.correct for o in adj), len(adj)),
        "adjust_proposed_pct": pct(sum(o.proposed_adjust for o in adj), len(adj)),
        "adjust_blocked_by_gateway_pct": pct(sum(o.proposed_adjust and o.final != "ADJUST_SETPOINT"
                                                 for o in adj), len(adj)),
        # Ajuste ejecutado pero en la dirección o el tag equivocados (p. ej. bajar la amina).
        "adjust_wrong_direction_pct": pct(sum(o.final == "ADJUST_SETPOINT" and not o.correct
                                              for o in adj), len(adj)),
        "hold_specificity_pct": pct(sum(o.correct for o in hold), len(hold)),
        "false_adjust_pct": pct(sum(o.final == "ADJUST_SETPOINT" for o in hold), len(hold)),
        "alarm_on_calm_pct": pct(sum(o.final == "RAISE_ALARM" for o in hold), len(hold)),
        "near_miss_false_adjust_pct": pct(sum(o.final == "ADJUST_SETPOINT" for o in near), len(near)),
        "near_miss_proposed_adjust_pct": pct(sum(o.proposed_adjust for o in near), len(near)),
        "balanced_accuracy_pct": (round((pct(sum(o.correct for o in adj), len(adj))
                                         + pct(sum(o.correct for o in hold), len(hold))) / 2, 1)
                                  if adj and hold else None),
        "citation_or_premise_warning_pct": pct(sum(bool(o.warnings) for o in outcomes), len(outcomes)),
        "failsafe_pct": pct(sum(o.tier == "failsafe" for o in outcomes), len(outcomes)),
        "latency_ms_p50": lat[len(lat) // 2] if lat else None,
        "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
    }
