"""Tests del PdMAgent: estimación de RUL, umbrales de degradación y datos de sensores problemáticos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from src.agents.pdm_agent import (
    DEFAULT_CRITICAL_RUL_HOURS,
    DEFAULT_WARNING_RUL_HOURS,
    ConditionSeries,
    HealthStatus,
    PdMAgent,
)
from src.tools import build_default_registry


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def agent(registry):
    return PdMAgent(registry, critical_rul_hours=24.0, warning_rul_hours=168.0)


def _series(metric, timestamps, measurements, failure_threshold):
    return ConditionSeries(
        metric=metric,
        timestamps=timestamps,
        measurements=measurements,
        failure_threshold=failure_threshold,
    )


# --------------------------------------------------------------------------- #
# Estimación de RUL frente a umbrales críticos de degradación
# --------------------------------------------------------------------------- #


def test_diagnose_flags_critical_when_rul_below_critical_threshold(agent):
    # Vibración sube 1.0 mm/s/h, falla a 50; en t=10 quedan 10h para el umbral -> CRITICAL (<=24h).
    series = [_series("vibration", [0, 1, 2, 3], [10, 11, 12, 13], failure_threshold=23)]

    diagnosis = agent.diagnose("SAG-01", series)

    assert diagnosis.overall_status == HealthStatus.CRITICAL
    assert diagnosis.bottleneck_metric == "vibration"
    assert diagnosis.recommended_maintenance_window_hours == pytest.approx(10.0, rel=1e-6)
    assert "mantenimiento de inmediato" in diagnosis.summary


def test_diagnose_flags_degrading_when_rul_between_warning_and_critical(agent):
    # Temperatura sube 1°C/h desde 20; falla a 120 -> en t=3, RUL = 97h (entre 24h y 168h) -> DEGRADING.
    series = [_series("temperature", [0, 1, 2, 3], [20, 21, 22, 23], failure_threshold=120)]

    diagnosis = agent.diagnose("PUMP-07", series)

    assert diagnosis.overall_status == HealthStatus.DEGRADING
    assert diagnosis.metric_diagnoses[0].trend == "increasing"


def test_diagnose_is_healthy_when_rul_beyond_warning_threshold(agent):
    # Carga sube muy lentamente; RUL muy por encima de 168h -> HEALTHY.
    series = [_series("load", [0, 1, 2, 3], [50.0, 50.01, 50.02, 50.03], failure_threshold=90.0)]

    diagnosis = agent.diagnose("CAEX-12", series)

    assert diagnosis.overall_status == HealthStatus.HEALTHY
    assert diagnosis.recommended_maintenance_window_hours > 168.0


def test_diagnose_is_healthy_when_metric_trends_away_from_failure(agent):
    # La medición decrece, alejándose del umbral de falla -> calculate_rul no proyecta falla (None).
    series = [_series("vibration", [0, 1, 2, 3], [40, 30, 20, 10], failure_threshold=100)]

    diagnosis = agent.diagnose("SAG-02", series)

    assert diagnosis.overall_status == HealthStatus.HEALTHY
    assert diagnosis.metric_diagnoses[0].rul_estimate is None
    assert diagnosis.metric_diagnoses[0].trend == "decreasing"


def test_diagnose_picks_the_most_urgent_metric_as_bottleneck(agent):
    critical_vibration = _series("vibration", [0, 1, 2, 3], [10, 11, 12, 13], failure_threshold=23)
    healthy_load = _series("load", [0, 1, 2, 3], [50.0, 50.01, 50.02, 50.03], failure_threshold=90.0)

    diagnosis = agent.diagnose("SAG-03", [critical_vibration, healthy_load])

    assert diagnosis.overall_status == HealthStatus.CRITICAL
    assert diagnosis.bottleneck_metric == "vibration"
    assert len(diagnosis.metric_diagnoses) == 2


def test_constructor_rejects_inverted_thresholds(registry):
    with pytest.raises(ValueError):
        PdMAgent(registry, critical_rul_hours=200.0, warning_rul_hours=24.0)


# --------------------------------------------------------------------------- #
# Datos de sensores incompletos
# --------------------------------------------------------------------------- #


def test_diagnose_raises_on_empty_series_list(agent):
    with pytest.raises(ValueError):
        agent.diagnose("SAG-04", [])


def test_diagnose_marks_single_reading_metric_as_insufficient_data(agent):
    series = [_series("vibration", [0.0], [10.0], failure_threshold=50.0)]

    diagnosis = agent.diagnose("SAG-05", series)

    assert diagnosis.overall_status == HealthStatus.INSUFFICIENT_DATA
    assert diagnosis.metric_diagnoses[0].status == HealthStatus.INSUFFICIENT_DATA
    assert diagnosis.metric_diagnoses[0].confidence == 0.0
    assert diagnosis.recommended_maintenance_window_hours is None


def test_diagnose_mismatched_lengths_is_rejected_at_construction():
    with pytest.raises(Exception):
        ConditionSeries(metric="vibration", timestamps=[0, 1, 2], measurements=[10, 20], failure_threshold=50)


def test_diagnose_degrades_gracefully_when_one_metric_lacks_data(agent):
    # Una métrica con datos insuficientes no debe abortar el diagnóstico de las demás.
    insufficient = _series("vibration", [0.0], [10.0], failure_threshold=50.0)
    critical = _series("temperature", [0, 1, 2, 3], [20, 30, 40, 50], failure_threshold=60)

    diagnosis = agent.diagnose("SAG-06", [insufficient, critical])

    assert len(diagnosis.metric_diagnoses) == 2
    vibration_diag = next(d for d in diagnosis.metric_diagnoses if d.metric == "vibration")
    temperature_diag = next(d for d in diagnosis.metric_diagnoses if d.metric == "temperature")
    assert vibration_diag.status == HealthStatus.INSUFFICIENT_DATA
    assert temperature_diag.status == HealthStatus.CRITICAL
    # El diagnóstico global debe reflejar la métrica sí utilizable, no quedar bloqueado por la otra.
    assert diagnosis.overall_status == HealthStatus.CRITICAL
    assert diagnosis.bottleneck_metric == "temperature"


# --------------------------------------------------------------------------- #
# Datos de sensores fuera de rango
# --------------------------------------------------------------------------- #


def test_diagnose_flags_out_of_range_readings_without_crashing(agent):
    # 500 mm/s RMS de vibración es físicamente implausible (rango: 0-50).
    series = [_series("vibration", [0, 1, 2, 3], [10, 12, 500, 14], failure_threshold=100)]

    diagnosis = agent.diagnose("SAG-07", series)

    metric_diag = diagnosis.metric_diagnoses[0]
    assert metric_diag.has_out_of_range_readings is True
    assert any("fuera del rango" in note for note in metric_diag.notes)


def test_out_of_range_readings_reduce_confidence_relative_to_clean_series(registry):
    agent = PdMAgent(registry)
    clean = _series("vibration", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 100)
    dirty = _series("vibration", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 500, 18, 19], 100)

    clean_diag = agent.diagnose("SAG-08", [clean]).metric_diagnoses[0]
    dirty_diag = agent.diagnose("SAG-09", [dirty]).metric_diagnoses[0]

    assert dirty_diag.confidence < clean_diag.confidence


def test_unknown_metric_name_skips_range_check_without_error(agent):
    # Una métrica no registrada en PLAUSIBLE_RANGES simplemente no se valida por rango.
    series = [_series("acoustic_emission", [0, 1, 2, 3], [1000, 1100, 1200, 1300], failure_threshold=5000)]

    diagnosis = agent.diagnose("PUMP-08", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is False


def test_diagnose_handles_unregistered_rul_tool_as_insufficient_data():
    from src.tools import ToolRegistry

    empty_registry = ToolRegistry()
    agent = PdMAgent(empty_registry)
    series = [_series("vibration", [0, 1, 2, 3], [10, 11, 12, 13], failure_threshold=23)]

    diagnosis = agent.diagnose("SAG-10", series)

    assert diagnosis.overall_status == HealthStatus.INSUFFICIENT_DATA
    assert diagnosis.metric_diagnoses[0].status == HealthStatus.INSUFFICIENT_DATA
    assert "No se pudo calcular RUL" in diagnosis.metric_diagnoses[0].notes[0]


# --------------------------------------------------------------------------- #
# Clasificación de tendencia
# --------------------------------------------------------------------------- #


def test_trend_is_stable_on_exactly_flat_series(agent):
    series = [_series("load", [0, 1, 2, 3], [50.0, 50.0, 50.0, 50.0], failure_threshold=90.0)]

    diagnosis = agent.diagnose("PUMP-10", series)

    assert diagnosis.metric_diagnoses[0].trend == "stable"


def test_trend_is_increasing_on_clear_upward_slope(agent):
    series = [_series("temperature", [0, 1, 2], [10, 20, 30], failure_threshold=1000)]

    diagnosis = agent.diagnose("PUMP-11", series)

    assert diagnosis.metric_diagnoses[0].trend == "increasing"


def test_trend_is_decreasing_on_clear_downward_slope(agent):
    series = [_series("temperature", [0, 1, 2], [30, 20, 10], failure_threshold=1000)]

    diagnosis = agent.diagnose("PUMP-12", series)

    assert diagnosis.metric_diagnoses[0].trend == "decreasing"


# --------------------------------------------------------------------------- #
# Bordes exactos de los umbrales de RUL (crítico/advertencia)
# --------------------------------------------------------------------------- #


def _linear_series_with_rul(metric: str, target_rul: float) -> ConditionSeries:
    """Serie de 2 puntos (pendiente 1, intercepto 0) cuyo RUL calculado es exactamente `target_rul`."""
    return _series(metric, [0, 1], [0, 1], failure_threshold=target_rul + 1.0)


def test_classify_status_is_critical_exactly_at_critical_boundary(agent):
    # Prueba de caja blanca sobre la regla de borde en sí (<=), sin el ruido de
    # punto flotante que introduce ajustar una recta con np.polyfit.
    assert agent._classify_status(agent.critical_rul_hours) == HealthStatus.CRITICAL


def test_rul_just_above_critical_threshold_is_degrading(agent):
    diagnosis = agent.diagnose("BOUND-02", [_linear_series_with_rul("vibration", 24.5)])

    assert diagnosis.overall_status == HealthStatus.DEGRADING


def test_classify_status_is_degrading_exactly_at_warning_boundary(agent):
    assert agent._classify_status(agent.warning_rul_hours) == HealthStatus.DEGRADING


def test_rul_just_above_warning_threshold_is_healthy(agent):
    diagnosis = agent.diagnose("BOUND-04", [_linear_series_with_rul("vibration", 168.5)])

    assert diagnosis.overall_status == HealthStatus.HEALTHY


def test_rul_of_zero_when_already_at_failure_threshold_is_critical(agent):
    # delta_needed == 0: el equipo ya está en el umbral de falla ahora mismo.
    series = [_series("vibration", [0, 1], [0, 1], failure_threshold=1.0)]

    diagnosis = agent.diagnose("BOUND-05", series)

    assert diagnosis.metric_diagnoses[0].rul_estimate == 0.0
    assert diagnosis.overall_status == HealthStatus.CRITICAL


# --------------------------------------------------------------------------- #
# Fórmula de confianza: cantidad de muestras
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sample_count,expected_confidence",
    [(2, 0.2), (3, 0.3), (5, 0.5), (8, 0.8), (10, 1.0), (15, 1.0), (20, 1.0)],
)
def test_confidence_scales_with_clean_sample_count(agent, sample_count, expected_confidence):
    timestamps = list(range(sample_count))
    measurements = [10.0 + 0.1 * i for i in range(sample_count)]
    series = [_series("vibration", timestamps, measurements, failure_threshold=1000.0)]

    diagnosis = agent.diagnose("CONF-01", series)

    assert diagnosis.metric_diagnoses[0].confidence == pytest.approx(expected_confidence)


@pytest.mark.parametrize("out_of_range_count,expected_confidence", [(0, 1.0), (2, 0.8), (5, 0.5), (10, 0.0)])
def test_confidence_penalized_proportionally_to_out_of_range_count(agent, out_of_range_count, expected_confidence):
    measurements = [10.0] * (10 - out_of_range_count) + [500.0] * out_of_range_count
    series = [_series("vibration", list(range(10)), measurements, failure_threshold=1000.0)]

    diagnosis = agent.diagnose("CONF-02", series)

    assert diagnosis.metric_diagnoses[0].confidence == pytest.approx(expected_confidence)


# --------------------------------------------------------------------------- #
# Rangos físicamente plausibles: bordes y por métrica
# --------------------------------------------------------------------------- #


def test_vibration_exact_lower_bound_is_not_flagged(agent):
    series = [_series("vibration", [0, 1, 2, 3], [0.0, 5, 10, 15], failure_threshold=100)]

    diagnosis = agent.diagnose("RANGE-01", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is False


def test_vibration_exact_upper_bound_is_not_flagged(agent):
    series = [_series("vibration", [0, 1, 2, 3], [50.0, 5, 10, 15], failure_threshold=100)]

    diagnosis = agent.diagnose("RANGE-02", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is False


def test_vibration_just_below_lower_bound_is_flagged(agent):
    series = [_series("vibration", [0, 1, 2, 3], [-0.1, 5, 10, 15], failure_threshold=100)]

    diagnosis = agent.diagnose("RANGE-03", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is True


def test_vibration_just_above_upper_bound_is_flagged(agent):
    series = [_series("vibration", [0, 1, 2, 3], [50.1, 5, 10, 15], failure_threshold=100)]

    diagnosis = agent.diagnose("RANGE-04", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is True


def test_temperature_below_plausible_range_is_flagged(agent):
    series = [_series("temperature", [0, 1, 2, 3], [-40.1, 10, 20, 30], failure_threshold=1000)]

    diagnosis = agent.diagnose("RANGE-05", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is True


def test_temperature_above_plausible_range_is_flagged(agent):
    series = [_series("temperature", [0, 1, 2, 3], [300.1, 10, 20, 30], failure_threshold=1000)]

    diagnosis = agent.diagnose("RANGE-06", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is True


def test_load_below_plausible_range_is_flagged(agent):
    series = [_series("load", [0, 1, 2, 3], [-0.1, 10, 20, 30], failure_threshold=1000)]

    diagnosis = agent.diagnose("RANGE-07", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is True


def test_load_above_plausible_range_is_flagged(agent):
    series = [_series("load", [0, 1, 2, 3], [150.1, 10, 20, 30], failure_threshold=1000)]

    diagnosis = agent.diagnose("RANGE-08", series)

    assert diagnosis.metric_diagnoses[0].has_out_of_range_readings is True


# --------------------------------------------------------------------------- #
# Agregación multi-métrica
# --------------------------------------------------------------------------- #


def _critical_series(metric: str = "vibration") -> ConditionSeries:
    return _series(metric, [0, 1], [0, 1], failure_threshold=11.0)  # RUL = 10.0 -> CRITICAL, con margen seguro


def _degrading_series(metric: str = "temperature") -> ConditionSeries:
    return _series(metric, [0, 1], [0, 1], failure_threshold=101.0)  # RUL = 100.0 -> DEGRADING


def _healthy_series(metric: str = "load") -> ConditionSeries:
    return _series(metric, [0, 1], [0, 1], failure_threshold=1001.0)  # RUL = 1000.0 -> HEALTHY


def _insufficient_series(metric: str = "vibration") -> ConditionSeries:
    return _series(metric, [0.0], [10.0], failure_threshold=50.0)


def test_aggregation_all_healthy_is_healthy(agent):
    diagnosis = agent.diagnose(
        "AGG-01", [_healthy_series("vibration"), _healthy_series("temperature"), _healthy_series("load")]
    )

    assert diagnosis.overall_status == HealthStatus.HEALTHY


def test_aggregation_one_critical_among_three_wins(agent):
    diagnosis = agent.diagnose(
        "AGG-02", [_healthy_series("load"), _degrading_series("temperature"), _critical_series("vibration")]
    )

    assert diagnosis.overall_status == HealthStatus.CRITICAL
    assert diagnosis.bottleneck_metric == "vibration"


def test_aggregation_degrading_beats_healthy_when_no_critical(agent):
    diagnosis = agent.diagnose("AGG-03", [_healthy_series("load"), _degrading_series("temperature")])

    assert diagnosis.overall_status == HealthStatus.DEGRADING
    assert diagnosis.bottleneck_metric == "temperature"


def test_aggregation_ignores_insufficient_metric_when_others_are_valid(agent):
    diagnosis = agent.diagnose("AGG-04", [_insufficient_series("vibration"), _healthy_series("load")])

    assert diagnosis.overall_status == HealthStatus.HEALTHY
    assert len(diagnosis.metric_diagnoses) == 2


def test_aggregation_all_insufficient_is_insufficient(agent):
    diagnosis = agent.diagnose(
        "AGG-05", [_insufficient_series("vibration"), _insufficient_series("temperature")]
    )

    assert diagnosis.overall_status == HealthStatus.INSUFFICIENT_DATA
    assert diagnosis.overall_confidence == 0.0


def test_aggregation_handles_duplicate_metric_names_without_crashing(agent):
    diagnosis = agent.diagnose("AGG-06", [_critical_series("vibration"), _healthy_series("vibration")])

    assert len(diagnosis.metric_diagnoses) == 2
    assert diagnosis.overall_status == HealthStatus.CRITICAL


def test_asset_id_is_preserved_in_the_diagnosis(agent):
    diagnosis = agent.diagnose("SAG-MILL-99", [_healthy_series()])

    assert diagnosis.asset_id == "SAG-MILL-99"


# --------------------------------------------------------------------------- #
# Contenido del resumen en texto plano
# --------------------------------------------------------------------------- #


def test_summary_mentions_immediate_maintenance_when_critical(agent):
    diagnosis = agent.diagnose("SUM-01", [_critical_series()])
    assert "inmediato" in diagnosis.summary


def test_summary_mentions_preventive_window_when_degrading(agent):
    diagnosis = agent.diagnose("SUM-02", [_degrading_series()])
    assert "preventivo" in diagnosis.summary


def test_summary_mentions_no_degradation_when_healthy(agent):
    diagnosis = agent.diagnose("SUM-03", [_healthy_series()])
    assert "sin tendencia de degradación" in diagnosis.summary


def test_summary_mentions_insufficient_data_when_insufficient(agent):
    diagnosis = agent.diagnose("SUM-04", [_insufficient_series()])
    assert "insuficientes" in diagnosis.summary


# --------------------------------------------------------------------------- #
# Validación de ConditionSeries
# --------------------------------------------------------------------------- #


def test_condition_series_rejects_unknown_extra_fields():
    with pytest.raises(ValidationError):
        ConditionSeries(
            metric="vibration",
            timestamps=[0, 1],
            measurements=[10, 20],
            failure_threshold=50,
            extra_field="not allowed",
        )


def test_condition_series_requires_failure_threshold():
    with pytest.raises(ValidationError):
        ConditionSeries(metric="vibration", timestamps=[0, 1], measurements=[10, 20])


def test_condition_series_rejects_empty_metric_name():
    with pytest.raises(ValidationError):
        ConditionSeries(metric="", timestamps=[0, 1], measurements=[10, 20], failure_threshold=50)


def test_condition_series_rejects_non_numeric_measurement():
    with pytest.raises(ValidationError):
        ConditionSeries(
            metric="vibration", timestamps=[0, 1], measurements=[{"bad": "value"}, 20], failure_threshold=50
        )


def test_condition_series_allows_both_series_empty(agent):
    # timestamps y measurements vacíos (longitudes iguales) construyen bien; el agente los marca insuficientes.
    series = ConditionSeries(metric="vibration", timestamps=[], measurements=[], failure_threshold=50)

    diagnosis = agent.diagnose("EMPTY-01", [series])

    assert diagnosis.metric_diagnoses[0].status == HealthStatus.INSUFFICIENT_DATA


def test_condition_series_allows_negative_timestamps(agent):
    # No se exige orden ni positividad de los timestamps, solo longitudes iguales.
    series = [_series("vibration", [-2, -1], [10, 20], failure_threshold=100)]

    diagnosis = agent.diagnose("NEG-01", series)

    assert diagnosis.metric_diagnoses[0].status != HealthStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# Validación del constructor de PdMAgent
# --------------------------------------------------------------------------- #


def test_constructor_rejects_negative_critical_threshold(registry):
    with pytest.raises(ValueError):
        PdMAgent(registry, critical_rul_hours=-1.0, warning_rul_hours=100.0)


def test_constructor_rejects_negative_warning_threshold(registry):
    with pytest.raises(ValueError):
        PdMAgent(registry, critical_rul_hours=10.0, warning_rul_hours=-1.0)


def test_constructor_allows_equal_critical_and_warning_thresholds(registry):
    agent = PdMAgent(registry, critical_rul_hours=48.0, warning_rul_hours=48.0)

    assert agent.critical_rul_hours == agent.warning_rul_hours == 48.0


def test_constructor_uses_default_thresholds_when_not_provided(registry):
    agent = PdMAgent(registry)

    assert agent.critical_rul_hours == DEFAULT_CRITICAL_RUL_HOURS
    assert agent.warning_rul_hours == DEFAULT_WARNING_RUL_HOURS
