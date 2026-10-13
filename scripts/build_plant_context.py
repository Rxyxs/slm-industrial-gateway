"""Construye los artefactos de contexto a partir del dataset real de la planta de flotación.

1. Historia horaria: promedio por hora de las lecturas de 20 s (el timestamp del archivo
   solo trae la hora). Una hora de sílice cuenta como resultado de laboratorio solo si no
   cambia dentro de la hora y no está en un tramo recto entre dos valores (interpolación).
   Los tramos se cortan en huecos de más de una hora; ningún rezago cruza un hueco.
2. Soft sensor: Ridge sobre variables de proceso + último resultado de laboratorio
   conocido con 2 h de retraso (el diseño que en el repo flotation-plant-silica-soft-sensor
   fue el único que le ganó a la persistencia). Se entrena en orden temporal con el 70%
   inicial y se evalúa en el 30% final; los coeficientes se exportan a JSON para que el
   gateway lo evalúe sin scikit-learn.
3. Casos de utilidad: horas reales del 30% final (fuera de muestra para el soft sensor),
   etiquetadas por la regla operativa documentada en `src/gateway/policy.py`. Las horas
   ambiguas no se usan.

    python scripts/build_plant_context.py --csv data/flotation/MiningProcess_Flotation_Plant_Database.csv
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gateway.context import HourRecord, LabResult, build_context  # noqa: E402
from src.gateway.policy import label_case  # noqa: E402
from src.gateway.soft_sensor import LAB_DELAY_H, SoftSensor  # noqa: E402
from src.gateway.tags import TAG_REGISTRY  # noqa: E402

TARGET = "% Silica Concentrate"
LINEAR_TOL = 1e-6
TRAIN_FRAC = 0.7
WINDOW_H = 8
# El soft sensor usa el resultado de 24 h atrás: cada caso guarda 48 h de laboratorio (cubre huecos entre análisis).
LAB_HISTORY_H = 48
# Variables del registro que el soft sensor usa como entrada (todas leídas en tiempo real).
SENSOR_TAGS = ["starch_flow", "amina_flow", "ore_pulp_flow", "ore_pulp_ph", "ore_pulp_density",
               "column_01_air_flow", "column_01_level", "iron_feed_pct", "silica_feed_pct"]
LAB_LAGS = (LAB_DELAY_H, LAB_DELAY_H + 1, LAB_DELAY_H + 2, 24)


def hourly(csv: Path) -> pl.DataFrame:
    raw = pl.read_csv(csv, infer_schema=False)
    numeric = [c for c in raw.columns if c != "date"]
    raw = raw.with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        *[pl.col(c).str.replace(",", ".", literal=True).cast(pl.Float64) for c in numeric],
    )
    cols = {spec.source_column: tag for tag, spec in TAG_REGISTRY.items()}
    hours = (raw.group_by("date").agg(
        *[pl.col(c).mean().alias(t) for c, t in cols.items()],
        (pl.col(TARGET).max() - pl.col(TARGET).min()).alias("_silica_within"),
    ).sort("date"))
    v = hours["silica_concentrate_pct"].to_numpy()
    straight = np.zeros(len(v), dtype=bool)
    second = np.abs(np.diff(v, 2)) < LINEAR_TOL
    moving = np.abs(np.diff(v)) > LINEAR_TOL
    straight[1:-1] = second & moving[1:]
    gap = hours["date"].diff().dt.total_hours().fill_null(1) > 1
    return hours.with_columns(
        measured=(pl.col("_silica_within") < LINEAR_TOL) & pl.Series(~straight),
        segment=pl.Series(gap.cast(pl.Int32).cum_sum()),
    ).drop("_silica_within")


def lab_lags(df: pl.DataFrame) -> pl.DataFrame:
    last = pl.when(pl.col("measured")).then(pl.col("silica_concentrate_pct")).otherwise(None)
    df = df.with_columns(last.forward_fill().over("segment").alias("_last"))
    df = df.with_columns([pl.col("_last").shift(lag).over("segment").alias(f"lab_lag{lag}") for lag in LAB_LAGS])
    return df.drop("_last")


def to_records(df: pl.DataFrame) -> list[HourRecord]:
    out = []
    for row in df.iter_rows(named=True):
        out.append(HourRecord(ts=row["date"], values={t: row[t] for t in TAG_REGISTRY
                                                       if t != "silica_concentrate_pct"}))
    return out


def lab_results(df: pl.DataFrame) -> list[LabResult]:
    """Resultados de laboratorio tal como llegan a planta: LAB_DELAY_H horas después de la muestra."""
    m = df.filter(pl.col("measured"))
    return [LabResult(sampled_at=r["date"], silica_pct=r["silica_concentrate_pct"], delay_h=LAB_DELAY_H)
            for r in m.iter_rows(named=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--sensor-out", type=Path, default=Path("src/gateway/models/silica_soft_sensor.json"))
    ap.add_argument("--cases-out", type=Path, default=Path("data/evals/utility_cases.json"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = lab_lags(hourly(args.csv))
    print(f"{len(df)} hours, {df['measured'].sum()} with a lab result, "
          f"{df['segment'].n_unique()} continuous segments")

    # --- soft sensor, orden temporal ---
    feats = SENSOR_TAGS + [f"lab_lag{l}" for l in LAB_LAGS]
    usable = df.filter(pl.col("measured")).drop_nulls(subset=feats)
    cut = int(len(usable) * TRAIN_FRAC)
    split_ts = usable["date"][cut]
    train, test = usable[:cut], usable[cut:]
    X_tr, y_tr = train.select(feats).to_numpy(), train["silica_concentrate_pct"].to_numpy()
    X_te, y_te = test.select(feats).to_numpy(), test["silica_concentrate_pct"].to_numpy()
    scaler = StandardScaler().fit(X_tr)
    model = Ridge(alpha=10.0).fit(scaler.transform(X_tr), y_tr)
    pred = model.predict(scaler.transform(X_te))
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    persist = test[f"lab_lag{LAB_DELAY_H}"].to_numpy()
    rmse_p = float(np.sqrt(np.mean((persist - y_te) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((y_tr.mean() - y_te) ** 2)))
    print(f"soft sensor (test, {len(test)} lab hours after {split_ts}): RMSE {rmse:.3f} | "
          f"persistence {rmse_p:.3f} | train mean {rmse_mean:.3f}")

    sensor = SoftSensor(
        features=feats, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        coef=model.coef_.tolist(), intercept=float(model.intercept_), rmse=round(rmse, 3),
        meta={"trained_on": f"{len(train)} lab hours up to {split_ts}", "test_hours": len(test),
              "rmse_persistence": round(rmse_p, 3), "rmse_train_mean": round(rmse_mean, 3),
              "lab_delay_h": LAB_DELAY_H, "source": "Kaggle: Quality Prediction in a Mining Process"},
    )
    args.sensor_out.parent.mkdir(parents=True, exist_ok=True)
    args.sensor_out.write_text(json.dumps(sensor.to_dict(), indent=1))

    # --- casos de utilidad en el 30% final ---
    records = to_records(df)
    labs = lab_results(df)
    by_ts = {r.ts: i for i, r in enumerate(records)}
    seg = dict(zip(df["date"].to_list(), df["segment"].to_list()))
    candidates: dict[str, list[dict]] = {"ADJUST": [], "HOLD": [], "HOLD_NEAR_MISS": []}
    for row in df.filter(pl.col("date") >= split_ts).iter_rows(named=True):
        i = by_ts[row["date"]]
        window = [r for r in records[max(0, i - WINDOW_H + 1): i + 1] if seg[r.ts] == seg[row["date"]]]
        if len(window) < WINDOW_H:
            continue
        ctx = build_context(window, labs, sensor)
        label = label_case(ctx, sensor.rmse)
        if label is None:
            continue
        candidates[label].append({
            "ts": row["date"].isoformat(), "expected": label,
            "telemetry": [{"tag": t, "value": round(v, 4)} for t, v in window[-1].values.items()],
            "history": [{"ts": r.ts.isoformat(), "values": {k: round(v, 4) for k, v in r.values.items()}}
                        for r in window],
            "lab_results": [{"sampled_at": l.sampled_at.isoformat(), "silica_pct": l.silica_pct}
                            for l in labs if l.available_at <= row["date"]
                            and l.sampled_at >= row["date"] - timedelta(hours=LAB_HISTORY_H)],
            "silica_measured_this_hour": row["silica_concentrate_pct"] if row["measured"] else None,
        })
    rng = np.random.default_rng(args.seed)
    cases = []
    # Clases balanceadas: tantas horas de cada clase como ADJUST haya (hasta --per-class).
    n = min(args.per_class, *(len(v) for v in candidates.values()))
    for label, items in candidates.items():
        print(f"{label}: {len(items)} eligible hours, {n} sampled")
        idx = rng.choice(len(items), size=n, replace=False)
        cases += [items[k] for k in sorted(idx)]
    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    args.cases_out.write_text(json.dumps({
        "source": "Kaggle: Quality Prediction in a Mining Process (CC0-1.0), hours after "
                  f"{split_ts} (out of sample for the soft sensor)",
        "rule": "src/gateway/policy.py", "window_h": WINDOW_H, "seed": args.seed, "cases": cases,
    }, indent=1))
    print(f"wrote {args.sensor_out} and {len(cases)} cases to {args.cases_out}")


if __name__ == "__main__":
    main()
