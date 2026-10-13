"""Contexto operativo: hechos, soft sensor, regla de referencia, citas y schema acotado."""

import json
from datetime import datetime, timedelta

import pytest

from src.gateway.context import ContextStore, HourRecord, LabResult, build_context, lab_as_of
from src.gateway.policy import label_case
from src.gateway.prompts import few_shot_examples
from src.gateway.router import InferenceRouter
from src.gateway.schemas import InferenceRequest, model_output_json_schema, setpoint_bounds
from src.gateway.soft_sensor import SoftSensor
from src.gateway.utility_eval import load_cases
from src.gateway.validation import check_citations, parse_action
from tests.gateway.conftest import FakeBackend, action_json

T0 = datetime(2017, 8, 1, 0)
BASE = {"starch_flow": 3000.0, "amina_flow": 500.0, "ore_pulp_flow": 398.0, "ore_pulp_ph": 9.9,
        "ore_pulp_density": 1.68, "column_01_air_flow": 280.0, "column_01_level": 520.0,
        "iron_feed_pct": 56.0, "silica_feed_pct": 14.0}


def window(n=8, **last):
    out = []
    for h in range(n):
        v = dict(BASE, amina_flow=480.0 + 2.5 * h)
        if h == n - 1:
            v.update(last)
        out.append(HourRecord(T0 + timedelta(hours=h), v))
    return out


def labs(*values, now=T0 + timedelta(hours=7)):
    """values del más antiguo al más reciente; el último se tomó 2 h antes de `now`."""
    n = len(values)
    return [LabResult(now - timedelta(hours=2 + (n - 1 - i)), v) for i, v in enumerate(values)]


class Fixed:
    def __init__(self, value, rmse=0.81):
        self.value, self.rmse, self.features = value, rmse, []

    def predict(self, x):
        return self.value


def test_facts_have_semantic_ids_and_exact_numbers():
    ctx = build_context(window(), labs(3.9, 4.4, 4.7), Fixed(3.8))
    assert ctx.ids[:4] == ["lab_last", "lab_prev", "lab_streak", "soft_sensor"]
    assert ctx.get("lab_last").value == 4.7 and ctx.get("lab_last").flag == "high"
    assert ctx.get("lab_streak").value == 2 and ctx.get("lab_streak").flag == "sustained_high"
    amina = ctx.get("amina_flow")
    assert amina.value == 497.5 and "trend +2.5/h" in amina.text and "room to raise 174.3" in amina.text
    assert "[ore_pulp_ph]" in ctx.render()


def test_lab_result_is_invisible_until_the_lab_returns_it():
    now = T0 + timedelta(hours=7)
    sampled_1h_ago = [LabResult(now - timedelta(hours=1), 5.0)]   # llega en 1 h más
    ctx = build_context(window(), sampled_1h_ago, None)
    assert ctx.get("lab_last") is None
    assert lab_as_of(sampled_1h_ago, now) is not None  # existe, pero no está disponible


def test_trend_window_and_ranges():
    ctx = build_context(window(ore_pulp_ph=10.9), labs(2.0, 2.1), None)
    assert ctx.get("ore_pulp_ph").flag == "out_of_range"
    assert ctx.get("ore_pulp_density").flag == ""


def test_policy_labels():
    assert label_case(build_context(window(), labs(4.3, 4.6), Fixed(3.6)), 0.81) == "ADJUST"
    assert label_case(build_context(window(), labs(2.0, 2.2), Fixed(2.1)), 0.81) == "HOLD"
    # Una sola muestra alta, o sílice alta pero bajando: el procedimiento dice esperar.
    assert label_case(build_context(window(), labs(2.9, 4.6), Fixed(3.6)), 0.81) == "HOLD_NEAR_MISS"
    assert label_case(build_context(window(), labs(5.0, 4.4), Fixed(3.6)), 0.81) == "HOLD_NEAR_MISS"
    # Dos altas, sin bajar, pero el soft sensor contradice al laboratorio: ambiguo.
    assert label_case(build_context(window(), labs(4.3, 4.6), Fixed(3.0)), 0.81) is None
    # Sin holgura de amina.
    assert label_case(build_context(window(amina_flow=665.0), labs(4.3, 4.6), Fixed(3.6)), 0.81) is None
    # pH fuera de rango manda sobre todo lo demás.
    assert label_case(build_context(window(ore_pulp_ph=10.9), labs(4.3, 4.6), Fixed(3.6)), 0.81) is None


def test_soft_sensor_artifact_beats_persistence_and_predicts():
    s = SoftSensor.load()
    assert s.rmse < s.meta["rmse_persistence"] < s.meta["rmse_train_mean"]
    x = {f: m for f, m in zip(s.features, s.mean)}  # en la media: predice el intercepto
    assert s.predict(x) == pytest.approx(s.intercept)
    x.pop(s.features[0])
    assert s.predict(x) is None


@pytest.mark.parametrize("evidence, reason, action, reject, warn", [
    (["lab_last", "lab_prev", "amina_flow"], "SILICA_HIGH", "ADJUST_SETPOINT", False, False),
    (["lab_last", "lab_prev"], "SILICA_HIGH", "ADJUST_SETPOINT", False, True),   # omite el tag: aviso
    (["amina_flow"], "SILICA_HIGH", "ADJUST_SETPOINT", True, True),              # motivo sin respaldo
    (["ore_pulp_ph"], "PH_OUT_OF_RANGE", "RAISE_ALARM", False, True),            # pH en rango: aviso
    (["made_up"], "SILICA_HIGH", "ADJUST_SETPOINT", True, False),                # cita inexistente
])
def test_citation_check(evidence, reason, action, reject, warn):
    ctx = build_context(window(), labs(4.3, 4.6), Fixed(3.6))
    extra = {"target_tag": "amina_flow", "new_value": 515} if action == "ADJUST_SETPOINT" else {}
    a = parse_action(action_json(action=action, reason_code=reason, evidence=evidence, **extra))
    r, w = check_citations(a, ctx)
    assert (r is not None) is reject and (w is not None) is warn


def test_bounded_schema_makes_unsafe_setpoints_unrepresentable():
    req = InferenceRequest(asset_id="a", telemetry=[
        {"tag": "amina_flow", "value": 557.4}, {"tag": "column_01_level", "value": 320.0},
        {"tag": "starch_flow", "value": 3000.0, "quality": "bad"}])
    bounds = setpoint_bounds(req.telemetry)
    assert bounds == {"amina_flow": (533, 582), "column_01_level": (316, 340)}  # nivel topado en low
    schema = model_output_json_schema(["lab_last", "amina_flow"], bounds)
    adj = schema["$defs"]["Adjust_amina_flow"]["properties"]
    assert adj["new_value"] == {"type": "integer", "minimum": 533, "maximum": 582,
                                "description": "new amina_flow setpoint, between 533 and 582"}
    assert adj["target_tag"] == {"const": "amina_flow"}
    assert "Adjust_starch_flow" not in schema["$defs"]  # lectura mala: no se puede ajustar
    assert schema["$defs"]["HoldOutput"]["properties"]["rationale"]["maxLength"] == 160
    assert list(schema["$defs"]["HoldOutput"]["properties"])[0] == "evidence"
    json.dumps(schema)


def test_few_shot_examples_are_valid_outputs_with_real_format():
    for user, answer in few_shot_examples():
        assert "[soft_sensor]" in user and "[amina_flow]" in user
        a = parse_action(answer)
        assert a.evidence and all(e in user for e in a.evidence)


def test_context_store_rolls_hours_and_keeps_labs():
    store = ContextStore(window_h=3)
    for h in range(5):
        for m in (0, 30):
            store.observe("a", T0 + timedelta(hours=h, minutes=m), {"amina_flow": 500.0 + h})
    w = store.window("a")
    # Horas 0-3 cerradas, la 4 sigue abierta; la ventana guarda las 3 más recientes.
    assert [r.values["amina_flow"] for r in w] == [501.0, 502.0, 503.0]
    assert w[-1].ts == T0 + timedelta(hours=3)
    store.add_lab("a", LabResult(T0, 4.2))
    assert store.labs("a")[0].silica_pct == 4.2 and store.labs("b") == []


async def test_router_uses_context_and_bounded_schema(settings, audit):
    case = next(c for c in load_cases() if c.expected == "ADJUST")
    edge = FakeBackend("edge", default=action_json(
        action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=round(case.amina_now) + 20,
        reason_code="SILICA_HIGH", evidence=["lab_last", "lab_prev", "amina_flow"]))
    seen = {}

    async def spy(system, user, schema, timeout_s, examples=()):
        seen.update(system=system, user=user, schema=schema, n_examples=len(examples))
        return await FakeBackend.generate(edge, system, user, schema, timeout_s)

    edge.generate = spy
    s = settings.__class__(**{**settings.__dict__, "prompt_mode": "context_fewshot"})
    resp = await InferenceRouter(s, [edge], audit).infer(case.request)
    assert resp.action.action == "ADJUST_SETPOINT" and resp.rejected_reason is None
    assert "[lab_last]" in seen["user"] and seen["n_examples"] == 5
    assert "Adjust_amina_flow" in seen["schema"]["$defs"]


async def test_router_rejects_adjust_with_invented_justification(settings, audit):
    case = next(c for c in load_cases() if c.expected == "HOLD")  # sílice <= 2,5%
    edge = FakeBackend("edge", default=action_json(
        action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=round(case.amina_now) + 20,
        reason_code="SILICA_HIGH", evidence=["lab_last", "amina_flow"]))
    resp = await InferenceRouter(settings, [edge], audit).infer(case.request)
    assert resp.action.action == "HOLD" and "SILICA_HIGH" in resp.rejected_reason


async def test_snapshot_mode_ignores_context(settings, audit):
    s = settings.__class__(**{**settings.__dict__, "prompt_mode": "snapshot"})
    case = load_cases()[0]
    router = InferenceRouter(s, [FakeBackend("edge", default=action_json())], audit)
    bundle = router.build_prompt(case.request)
    assert bundle.ctx is None and "FACTS" not in bundle.user and not bundle.examples


def test_utility_cases_reproduce_their_labels():
    sensor = SoftSensor.load()
    cases = load_cases()
    counts = {k: sum(c.expected == k for c in cases) for k in ("ADJUST", "HOLD", "HOLD_NEAR_MISS")}
    assert counts == {"ADJUST": 40, "HOLD": 40, "HOLD_NEAR_MISS": 40}
    for c in cases:
        assert label_case(c.context(sensor), sensor.rmse) == c.expected, c.ts


@pytest.mark.parametrize("values, state", [
    ((4.3, 4.6), "SUSTAINED_HIGH"), ((5.0, 4.4), "RECOVERING"),
    ((2.9, 4.6), "SINGLE_HIGH"), ((4.6, 3.0), "NORMAL"),
])
def test_derived_silica_state(values, state):
    ctx = build_context(window(), labs(*values), Fixed(3.6), derived=True)
    assert f"= {state}:" in ctx.get("silica_state").text
    assert build_context(window(), labs(*values), Fixed(3.6)).get("silica_state") is None


async def test_gated_mode_removes_adjust_when_procedure_says_wait(settings, audit):
    s = settings.__class__(**{**settings.__dict__, "prompt_mode": "context_gated"})
    router = InferenceRouter(s, [FakeBackend("edge", default=action_json())], audit)
    for label, allowed in (("ADJUST", True), ("HOLD_NEAR_MISS", False), ("HOLD", False)):
        case = next(c for c in load_cases() if c.expected == label)
        schema = router.build_prompt(case.request).schema
        adjust = [k for k in schema["$defs"] if k.startswith("Adjust_")]
        assert bool(adjust) is allowed, label
        assert "[silica_state]" in router.build_prompt(case.request).user
        if allowed:
            assert adjust == ["Adjust_amina_flow"]
            lo = schema["$defs"]["Adjust_amina_flow"]["properties"]["new_value"]["minimum"]
            assert lo > case.amina_now  # solo hacia arriba
