"""Soft sensor de sílice en concentrado, evaluado sin scikit-learn.

Es un Ridge sobre las variables de proceso de la hora actual más el último resultado de
laboratorio conocido (la planta recibe el análisis `LAB_DELAY_H` horas después de tomar la
muestra). Se entrena con `scripts/build_plant_context.py` y se exporta como coeficientes en
JSON; acá solo se aplica `(x - mean) / scale · coef + intercept`, así la imagen del gateway
no carga scikit-learn ni numpy.

La estimación viaja al modelo con su error (`rmse` medido fuera de muestra, en orden
temporal): un soft sensor de esta planta no es preciso, y el SLM tiene que saberlo.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

LAB_DELAY_H = 2
DEFAULT_PATH = Path(__file__).resolve().parent / "models" / "silica_soft_sensor.json"


@dataclass(frozen=True)
class SoftSensor:
    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    rmse: float
    meta: dict = field(default_factory=dict)

    def predict(self, x: dict[str, float]) -> float | None:
        """None si falta alguna entrada: mejor no estimar que estimar con un hueco."""
        total = self.intercept
        for name, mu, sd, w in zip(self.features, self.mean, self.scale, self.coef):
            v = x.get(name)
            if v is None or not math.isfinite(v):
                return None
            total += (v - mu) / sd * w
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "SoftSensor":
        return cls(**json.loads(Path(path).read_text()))
