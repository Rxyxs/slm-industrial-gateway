"""Pruebas unitarias para el pipeline de fine-tuning QLoRA (src/training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.dataset_prep import (
    CONTROL_TOKENS,
    DEFAULT_DATASET_PATH,
    DatasetFormatError,
    IM_END,
    IM_START,
    ensure_control_tokens,
    format_chatml,
    load_raw_examples,
    load_sft_dataset,
    safe_tokenize,
    to_sft_examples,
)
from src.training.finetune import (
    QLoRATrainingConfig,
    build_training_arguments,
    load_base_model_and_tokenizer,
)


class FakeTokenizer:
    """Doble de prueba que imita la interfaz mínima de un tokenizador de Hugging Face."""

    def __init__(self, vocab=None, pad_token=None, eos_token="<eos>"):
        self._vocab = dict(vocab) if vocab else {}
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.add_special_tokens_calls: list[dict] = []
        self.call_kwargs: dict | None = None

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, mapping):
        self.add_special_tokens_calls.append(mapping)
        added = mapping.get("additional_special_tokens", [])
        for offset, token in enumerate(added):
            self._vocab[token] = len(self._vocab) + offset
        return len(added)

    def __call__(self, text, max_length=None, truncation=None, padding=None, return_attention_mask=None):
        self.call_kwargs = {
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
            "return_attention_mask": return_attention_mask,
        }
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        input_ids = list(range(len(tokens)))
        attention_mask = [1] * len(input_ids)
        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result


SAMPLE_MESSAGES = [
    {"role": "system", "content": " Eres un asistente experto en telemetría industrial. "},
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte?"},
]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def test_load_raw_examples_from_real_dataset_file():
    examples = load_raw_examples(DEFAULT_DATASET_PATH)
    assert len(examples) >= 30
    for example in examples:
        roles = [m["role"] for m in example["messages"]]
        assert "user" in roles
        assert "assistant" in roles


def test_load_raw_examples_missing_file(tmp_path):
    missing_path = tmp_path / "no_existe.json"
    with pytest.raises(FileNotFoundError):
        load_raw_examples(missing_path)


def test_load_raw_examples_rejects_empty_list(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        load_raw_examples(dataset_path)


def test_load_raw_examples_rejects_missing_messages_key(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"foo": "bar"}]), encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        load_raw_examples(dataset_path)


def test_load_raw_examples_rejects_missing_assistant_role(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = [{"messages": [{"role": "user", "content": "hola"}]}]
    dataset_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        load_raw_examples(dataset_path)


def test_load_raw_examples_rejects_invalid_role(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = [
        {
            "messages": [
                {"role": "user", "content": "hola"},
                {"role": "narrator", "content": "algo"},
                {"role": "assistant", "content": "respuesta"},
            ]
        }
    ]
    dataset_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        load_raw_examples(dataset_path)


def test_load_raw_examples_rejects_empty_content(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = [
        {
            "messages": [
                {"role": "user", "content": "   "},
                {"role": "assistant", "content": "respuesta"},
            ]
        }
    ]
    dataset_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        load_raw_examples(dataset_path)


def test_load_sft_dataset_builds_text_column():
    dataset = load_sft_dataset(DEFAULT_DATASET_PATH)
    assert dataset.num_rows >= 30
    assert "text" in dataset.column_names
    assert IM_START in dataset[0]["text"]
    assert IM_END in dataset[0]["text"]


# ---------------------------------------------------------------------------
# Formateo ChatML
# ---------------------------------------------------------------------------


def test_format_chatml_wraps_each_message_with_control_tokens():
    text = format_chatml(SAMPLE_MESSAGES)
    assert text.count(IM_START) == 3
    assert text.count(IM_END) == 3
    assert "<|im_start|>system\nEres un asistente experto en telemetría industrial.<|im_end|>" in text
    assert "<|im_start|>user\nHola<|im_end|>" in text
    assert "<|im_start|>assistant\nHola, ¿en qué puedo ayudarte?<|im_end|>" in text


def test_format_chatml_strips_surrounding_whitespace_from_content():
    text = format_chatml(SAMPLE_MESSAGES)
    assert " Eres un asistente" not in text
    assert "industrial. <|im_end|>" not in text


def test_to_sft_examples_returns_text_field():
    raw_examples = [{"messages": SAMPLE_MESSAGES}]
    sft_examples = to_sft_examples(raw_examples)
    assert sft_examples == [{"text": format_chatml(SAMPLE_MESSAGES)}]


def test_ensure_control_tokens_adds_missing_tokens():
    tokenizer = FakeTokenizer(vocab={"hola": 0})
    added = ensure_control_tokens(tokenizer)
    assert added == len(CONTROL_TOKENS)
    assert len(tokenizer.add_special_tokens_calls) == 1
    assert set(CONTROL_TOKENS) <= set(tokenizer.get_vocab())


def test_ensure_control_tokens_is_noop_when_already_present():
    vocab = {token: i for i, token in enumerate(CONTROL_TOKENS)}
    tokenizer = FakeTokenizer(vocab=vocab)
    added = ensure_control_tokens(tokenizer)
    assert added == 0
    assert tokenizer.add_special_tokens_calls == []


# ---------------------------------------------------------------------------
# Tokenización segura
# ---------------------------------------------------------------------------


def test_safe_tokenize_rejects_empty_text():
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError):
        safe_tokenize(tokenizer, "   ")


def test_safe_tokenize_sets_pad_token_from_eos_when_missing():
    tokenizer = FakeTokenizer(pad_token=None, eos_token="<eos>")
    safe_tokenize(tokenizer, "hola mundo")
    assert tokenizer.pad_token == "<eos>"


def test_safe_tokenize_truncates_with_expected_kwargs():
    tokenizer = FakeTokenizer()
    safe_tokenize(tokenizer, "una dos tres cuatro cinco", max_length=3)
    assert tokenizer.call_kwargs == {
        "max_length": 3,
        "truncation": True,
        "padding": False,
        "return_attention_mask": True,
    }


def test_safe_tokenize_raises_on_empty_output():
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError):
        safe_tokenize(tokenizer, "hola mundo", max_length=0)


# ---------------------------------------------------------------------------
# Inicialización de argumentos de entrenamiento (QLoRA)
# ---------------------------------------------------------------------------


def test_qlora_training_config_defaults_are_valid():
    config = QLoRATrainingConfig()
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.lora_r == 16
    assert len(config.lora_target_modules) > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"bnb_4bit_quant_type": "fp4"},
        {"load_in_4bit": False},
        {"lora_r": 0},
        {"lora_r": 512},
        {"lora_alpha": 0},
        {"lora_dropout": 1.0},
        {"per_device_train_batch_size": 0},
        {"learning_rate": 0.0},
        {"max_seq_length": 0},
    ],
)
def test_qlora_training_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        QLoRATrainingConfig(**overrides)


def test_build_training_arguments_reflects_config_values(tmp_path):
    config = QLoRATrainingConfig(
        output_dir=str(tmp_path / "out"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        max_seq_length=1024,
        seed=42,
    )
    args = build_training_arguments(config)

    assert args.output_dir == str(tmp_path / "out")
    assert args.num_train_epochs == 5
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 2
    assert args.learning_rate == 1e-4
    assert args.max_length == 1024
    assert args.seed == 42
    assert args.dataset_text_field == "text"
    assert args.packing is False


def test_load_base_model_and_tokenizer_raises_clear_error_without_unsloth():
    config = QLoRATrainingConfig()
    with pytest.raises(ImportError, match="Unsloth"):
        load_base_model_and_tokenizer(config)
