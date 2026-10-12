import json

import pytest

from src.gateway.schemas import InferenceRequest, control_action_json_schema, model_output_json_schema
from src.gateway.validation import (
    OutputRejected, check_physics, check_premise, parse_action, validate_output,
)
from tests.gateway.conftest import ADJUST_AMINA, action_json


def test_valid_hold_parses():
    a = parse_action(action_json())
    assert a.action == "HOLD" and a.target_tag is None


def test_valid_adjust_within_step_passes(plant_request):
    # 557.434 -> 545 es un paso de 12.4 < max_step 25
    raw = action_json(action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=545.0,
                      reason_code="SILICA_HIGH")
    action, reason = validate_output(raw, plant_request.telemetry)
    assert reason is None and action.new_value == 545.0


@pytest.mark.parametrize("raw, fragment", [
    ("Sure! Here is the JSON: {}", "not valid JSON"),
    ("[1, 2]", "must be an object"),
    (action_json(extra_field=1), "extra_field"),
    (action_json(action="SHUTDOWN_PLANT"), "action"),
    (action_json(action="ADJUST_SETPOINT"), "ADJUST_SETPOINT.target_tag"),
    # La variante HOLD no tiene campos de setpoint: ni siquiera en null.
    (action_json(action="HOLD", target_tag="amina_flow", new_value=500.0), "HOLD.target_tag"),
    (action_json(action="HOLD", target_tag=None), "HOLD.target_tag"),
    (action_json(action="ADJUST_SETPOINT", target_tag="ore_pulp_ph", new_value=9.8), "target_tag"),
    (action_json(action="ADJUST_SETPOINT", target_tag="reactor_core", new_value=1.0), "target_tag"),
    (action_json(confidence=3), "confidence"),  # salida real de qwen2.5:1.5b con confidence float
])
def test_shape_rejections(raw, fragment):
    with pytest.raises(OutputRejected) as exc:
        parse_action(raw)
    assert fragment in str(exc.value)


def test_out_of_range_setpoint_rejected(plant_request):
    a = parse_action(action_json(action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=900.0,
                                 reason_code="SILICA_HIGH"))
    with pytest.raises(OutputRejected, match="outside"):
        check_physics(a, plant_request.telemetry)


def test_step_too_large_rejected(plant_request):
    # 557.4 -> 600 está en rango pero es un paso de 42.6 > 25
    a = parse_action(action_json(action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=600.0,
                                 reason_code="SILICA_HIGH"))
    with pytest.raises(OutputRejected, match="max_step"):
        check_physics(a, plant_request.telemetry)


def test_adjust_without_current_reading_rejected():
    req = InferenceRequest(asset_id="a", telemetry=[{"tag": "ore_pulp_ph", "value": 9.9}])
    a = parse_action(ADJUST_AMINA)
    with pytest.raises(OutputRejected, match="no good-quality reading"):
        check_physics(a, req.telemetry)


def test_bad_quality_reading_does_not_bound_step():
    req = InferenceRequest(asset_id="a", telemetry=[
        {"tag": "amina_flow", "value": 515.0, "quality": "bad"}])
    with pytest.raises(OutputRejected, match="no good-quality reading"):
        check_physics(parse_action(ADJUST_AMINA), req.telemetry)


def test_long_rationale_is_trimmed_not_rejected():
    a = parse_action(action_json(rationale="word " * 100))
    assert len(a.rationale) == 200


def test_rejection_becomes_failsafe_hold(plant_request):
    action, reason = validate_output("not json", plant_request.telemetry)
    assert action.action == "HOLD" and action.confidence == "LOW"
    assert reason.startswith("not valid JSON")


def test_request_rejects_unknown_tag_and_nan():
    with pytest.raises(ValueError):
        InferenceRequest(asset_id="a", telemetry=[{"tag": "made_up", "value": 1}])
    with pytest.raises(ValueError):
        InferenceRequest(asset_id="a", telemetry=[{"tag": "amina_flow", "value": float("nan")}])


def test_guided_schema_is_a_discriminated_union_of_writable_tags():
    schema = model_output_json_schema()
    assert schema["discriminator"]["propertyName"] == "action"
    assert set(schema["$defs"]["WritableTag"]["enum"]) == {
        "starch_flow", "amina_flow", "column_01_air_flow", "column_01_level"}
    hold = schema["$defs"]["HoldOutput"]
    assert "target_tag" not in hold["properties"] and hold["additionalProperties"] is False
    assert schema["$defs"]["Confidence"]["enum"] == ["LOW", "MEDIUM", "HIGH"]
    json.dumps(schema)  # serializable para enviarlo al motor
    assert control_action_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize("reason_code, telemetry, contradicted", [
    ("PH_OUT_OF_RANGE", [{"tag": "ore_pulp_ph", "value": 10.13}], True),
    ("PH_OUT_OF_RANGE", [{"tag": "ore_pulp_ph", "value": 11.2}], False),
    ("SILICA_HIGH", [{"tag": "silica_concentrate_pct", "value": 1.3}], True),
    ("SILICA_HIGH", [{"tag": "silica_concentrate_pct", "value": 4.9}], False),
    ("SENSOR_QUALITY", [{"tag": "amina_flow", "value": 500}], True),
    ("SENSOR_QUALITY", [{"tag": "amina_flow", "value": 500, "quality": "bad"}], False),
    ("WITHIN_NORMAL", [{"tag": "ore_pulp_ph", "value": 11.2}], False),
])
def test_premise_check(reason_code, telemetry, contradicted):
    req = InferenceRequest(asset_id="a", telemetry=telemetry)
    a = parse_action(action_json(reason_code=reason_code))
    assert (check_premise(a, req.telemetry) is not None) is contradicted
