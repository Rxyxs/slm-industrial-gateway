"""Fine-tuning QLoRA (Unsloth + TRL SFTTrainer) para adaptación de dominio industrial/minero.

Requiere GPU con soporte CUDA, `unsloth` y `bitsandbytes` instalados para ejecutar el
entrenamiento real. La construcción de configuración, dataset y argumentos de
entrenamiento no requiere GPU y puede validarse en CPU (ver tests/test_training.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.training.dataset_prep import DEFAULT_DATASET_PATH, load_sft_dataset

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class QLoRATrainingConfig:
    """Hiperparámetros del pipeline QLoRA. No requiere GPU ni librerías pesadas para construirse."""

    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    dataset_path: str = str(DEFAULT_DATASET_PATH)
    max_seq_length: int = 2048
    output_dir: str = "outputs/qlora-industrial"
    merged_output_dir: str = "outputs/qlora-industrial-merged"

    # Cuantización QLoRA: NF4 a 4 bits con doble cuantización.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # Adaptadores LoRA.
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LORA_TARGET_MODULES)

    # Argumentos de entrenamiento.
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    optim: str = "adamw_8bit"
    seed: int = 3407

    def __post_init__(self) -> None:
        if self.bnb_4bit_quant_type != "nf4":
            raise ValueError("QLoRA requiere cuantización NF4 ('nf4').")
        if not self.load_in_4bit:
            raise ValueError("QLoRA requiere 'load_in_4bit=True'.")
        if not (0 < self.lora_r <= 256):
            raise ValueError("lora_r debe estar en el rango (0, 256].")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha debe ser positivo.")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout debe estar en el rango [0, 1).")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size debe ser positivo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo.")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length debe ser positivo.")


def build_training_arguments(config: QLoRATrainingConfig):
    """Construye el `SFTConfig` (TRL) con los argumentos de entrenamiento.

    La precisión mixta se resuelve según el hardware real disponible (bf16 en
    GPUs Ampere+, fp16 como respaldo en GPUs sin soporte bf16, ninguna en
    CPU) en vez de asumir GPU, para que valga tanto en el nodo de
    entrenamiento como en una máquina sin CUDA usada solo para validar la
    configuración.
    """
    import torch
    from trl import SFTConfig

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        optim=config.optim,
        seed=config.seed,
        max_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )


def load_base_model_and_tokenizer(config: QLoRATrainingConfig):
    """Carga el modelo base cuantizado en NF4 (4-bit) y su tokenizador vía Unsloth.

    Requiere GPU CUDA + `unsloth` + `bitsandbytes` instalados.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth no está instalado o no hay GPU CUDA disponible. Instala 'unsloth' y "
            "'bitsandbytes' en un entorno con GPU para ejecutar el entrenamiento QLoRA real."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,
    )
    return model, tokenizer


def attach_lora_adapters(model, config: QLoRATrainingConfig):
    """Agrega adaptadores LoRA al modelo cuantizado usando Unsloth."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias=config.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )


def build_trainer(model, tokenizer, train_dataset, config: QLoRATrainingConfig):
    """Construye el `SFTTrainer` de TRL para el entrenamiento QLoRA."""
    from trl import SFTTrainer

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=build_training_arguments(config),
    )


def save_lora_adapters(model, tokenizer, output_dir: str) -> None:
    """Guarda únicamente los adaptadores LoRA entrenados (sin fusionar con el modelo base)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def merge_and_save(model, tokenizer, merged_output_dir: str) -> None:
    """Fusiona los adaptadores LoRA con el modelo base y guarda los pesos combinados."""
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)


def run_training(config: QLoRATrainingConfig | None = None):
    """Ejecuta el pipeline completo: dataset, carga NF4, LoRA, SFTTrainer y guardado/merge."""
    config = config or QLoRATrainingConfig()

    train_dataset = load_sft_dataset(config.dataset_path)
    model, tokenizer = load_base_model_and_tokenizer(config)
    model = attach_lora_adapters(model, config)

    trainer = build_trainer(model, tokenizer, train_dataset, config)
    trainer.train()

    save_lora_adapters(model, tokenizer, config.output_dir)
    merge_and_save(model, tokenizer, config.merged_output_dir)

    return trainer


if __name__ == "__main__":
    run_training()
