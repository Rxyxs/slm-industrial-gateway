"""Verifica el pipeline de fine-tuning QLoRA (dataset, config, args de
entrenamiento) y deja constancia explícita de si el entrenamiento real corrió
o quedó bloqueado por falta de GPU -- nunca inventa loss/perplexity de una
corrida que no ocurrió.

`tests/test_training.py` ya cubre esto como pruebas unitarias (ver README,
sección "Pruebas"); este script arma el mismo chequeo como una corrida
reproducible de punta a punta y lo deja en outputs/reports/finetuning_metrics.json,
siguiendo la misma convención de scripts/generate_plots.py: separar
explícitamente lo verificado de lo que falta por correr en hardware real.

    python scripts/verify_finetuning_pipeline.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.dataset_prep import DEFAULT_DATASET_PATH, load_raw_examples, load_sft_dataset
from src.training.finetune import QLoRATrainingConfig, build_training_arguments, load_base_model_and_tokenizer

REPORTS_DIR = ROOT / "outputs" / "reports"
ADAPTER_DIR = ROOT / "outputs" / "adapters" / "industrial_qlora"


def check_dataset() -> dict:
    raw_examples = load_raw_examples(DEFAULT_DATASET_PATH)
    sft_dataset = load_sft_dataset(DEFAULT_DATASET_PATH)
    roles_per_example = [sorted({m["role"] for m in ex["messages"]}) for ex in raw_examples]
    return {
        "path": str(DEFAULT_DATASET_PATH),
        "n_examples": len(raw_examples),
        "n_sft_rows": sft_dataset.num_rows,
        "has_system_prompt_in_all_examples": all("system" in roles for roles in roles_per_example),
        "status": "ok",
    }


def check_config() -> dict:
    config = QLoRATrainingConfig()
    args = build_training_arguments(config)
    return {
        "model_name": config.model_name,
        "load_in_4bit": config.load_in_4bit,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_target_modules": list(config.lora_target_modules),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "learning_rate": config.learning_rate,
        "resolved_precision": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32 (sin GPU)"),
        "status": "ok",
    }


def check_environment() -> dict:
    import torch

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for package in ("unsloth", "bitsandbytes"):
        try:
            __import__(package)
            env[f"{package}_installed"] = True
        except ImportError:
            env[f"{package}_installed"] = False
    return env


def attempt_training() -> dict:
    """Intenta cargar el modelo base cuantizado. Sin GPU + unsloth, esto debe
    fallar con ImportError de forma explícita (mismo contrato que
    tests/test_training.py::test_load_base_model_and_tokenizer_raises_clear_error_without_unsloth) --
    nunca se reporta loss/perplexity de un entrenamiento que no llegó a correr."""
    config = QLoRATrainingConfig()
    try:
        load_base_model_and_tokenizer(config)
    except ImportError as exc:
        return {
            "status": "blocked_no_gpu",
            "stage_reached": "load_base_model_and_tokenizer",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "loss_metrics": None,
            "perplexity_before": None,
            "perplexity_after": None,
            "adapter_saved": False,
            "how_to_complete": (
                "Correr este mismo script (o src/training/finetune.py:run_training) en una "
                "maquina con GPU CUDA y 'unsloth'+'bitsandbytes' instalados (ver "
                "requirements.txt, seccion 'Fine-tuning QLoRA'). El adaptador quedaria en "
                f"{ADAPTER_DIR.relative_to(ROOT)} y este JSON se regeneraria con loss de "
                "entrenamiento/validacion real y perplexity inicial vs. final medidos, no estimados."
            ),
        }
    return {
        "status": "unexpected_success",
        "stage_reached": "load_base_model_and_tokenizer",
        "note": "el modelo cargo sin error pese a no haber GPU/unsloth detectados -- revisar manualmente",
    }


def main() -> None:
    print("[1/3] Verificando dataset sintetico industrial...")
    dataset_report = check_dataset()
    print(f"  {dataset_report['n_examples']} ejemplos ChatML validos en {dataset_report['path']}")

    print("[2/3] Verificando QLoRATrainingConfig y argumentos de entrenamiento...")
    config_report = check_config()
    print(f"  precision resuelta: {config_report['resolved_precision']}")

    print("[3/3] Intentando cargar el modelo base cuantizado (NF4)...")
    training_report = attempt_training()
    print(f"  {training_report['status']}")
    if training_report["status"] == "blocked_no_gpu":
        print(f"  {training_report['error_message']}")

    environment_report = check_environment()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "QLoRA fine-tuning sobre dataset sintetico industrial",
        "dataset": dataset_report,
        "training_config": config_report,
        "environment": environment_report,
        "training": training_report,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "finetuning_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado en: {out_path}")
    if training_report["status"] == "blocked_no_gpu":
        print(
            "\nAVISO: no se genero ningun adaptador LoRA ni metricas de loss/perplexity -- "
            "el entrenamiento real requiere GPU CUDA y no corrio. Ver 'how_to_complete' en el JSON."
        )


if __name__ == "__main__":
    main()
