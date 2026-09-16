"""Carga y preparación del dataset de dominio (ChatML) para fine-tuning QLoRA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CONTROL_TOKENS = (IM_START, IM_END)

VALID_ROLES = ("system", "user", "assistant")

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "domain_dataset" / "industrial_instructions.json"
)


class DatasetFormatError(ValueError):
    """Error de formato en un ejemplo del dataset ChatML."""


def load_raw_examples(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Carga y valida el dataset ChatML crudo (lista de {'messages': [...]}) desde un archivo JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or len(data) == 0:
        raise DatasetFormatError("El dataset debe ser una lista no vacía de ejemplos.")

    for index, example in enumerate(data):
        _validate_example(example, index=index)

    return data


def _validate_example(example: Any, index: int) -> None:
    if not isinstance(example, dict) or "messages" not in example:
        raise DatasetFormatError(f"Ejemplo {index}: debe contener la clave 'messages'.")

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise DatasetFormatError(f"Ejemplo {index}: 'messages' debe ser una lista no vacía.")

    roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
    if "user" not in roles or "assistant" not in roles:
        raise DatasetFormatError(
            f"Ejemplo {index}: debe incluir al menos un mensaje 'user' y uno 'assistant'."
        )

    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise DatasetFormatError(f"Ejemplo {index}: cada mensaje requiere 'role' y 'content'.")
        if message["role"] not in VALID_ROLES:
            raise DatasetFormatError(f"Ejemplo {index}: rol inválido '{message['role']}'.")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise DatasetFormatError(
                f"Ejemplo {index}: el contenido del mensaje '{message['role']}' no puede estar vacío."
            )


def format_chatml(messages: Iterable[dict[str, str]]) -> str:
    """Serializa mensajes {role, content} al formato ChatML delimitado por tokens de control."""
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"{IM_START}{role}\n{content}{IM_END}")
    return "\n".join(parts) + "\n"


def to_sft_examples(raw_examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convierte ejemplos crudos ChatML a registros {'text': ...} listos para SFTTrainer."""
    return [{"text": format_chatml(example["messages"])} for example in raw_examples]


def load_sft_dataset(path: str | Path = DEFAULT_DATASET_PATH):
    """Carga el dataset de dominio como un `datasets.Dataset` con una única columna 'text'."""
    from datasets import Dataset

    raw_examples = load_raw_examples(path)
    sft_examples = to_sft_examples(raw_examples)
    return Dataset.from_list(sft_examples)


def ensure_control_tokens(tokenizer) -> int:
    """Agrega `<|im_start|>`/`<|im_end|>` como tokens especiales si el tokenizador no los conoce.

    Retorna la cantidad de tokens nuevos añadidos, para que el llamador pueda
    redimensionar los embeddings del modelo (`resize_token_embeddings`) si corresponde.
    """
    vocab = tokenizer.get_vocab()
    missing = [token for token in CONTROL_TOKENS if token not in vocab]
    if not missing:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": missing})


def safe_tokenize(tokenizer, text: str, max_length: int = 2048) -> dict[str, list[int]]:
    """Tokeniza `text` de forma segura: valida contenido no vacío, trunca y asegura pad token."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No se puede tokenizar un texto vacío.")

    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token

    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )

    if len(encoded["input_ids"]) == 0:
        raise ValueError("La tokenización produjo una secuencia vacía.")

    return encoded
