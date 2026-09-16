#!/usr/bin/env python
"""Verifica y carga modelos GGUF cuantizados (Q4_K_M, Q8_0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine.llm_server import LLMServer, ModelLoadError  # noqa: E402

GGUF_MAGIC = b"GGUF"
SUPPORTED_QUANTIZATIONS = ("Q4_K_M", "Q8_0")


def verify_gguf(model_path: Path) -> None:
    """Verifica que el archivo exista y tenga la cabecera mágica GGUF."""
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {model_path}")

    with model_path.open("rb") as f:
        magic = f.read(4)

    if magic != GGUF_MAGIC:
        raise ValueError(f"{model_path} no es un archivo GGUF válido (cabecera: {magic!r}).")


def detect_quantization(model_path: Path) -> Optional[str]:
    """Detecta el tipo de cuantización a partir del nombre de archivo."""
    name = model_path.stem.upper()
    for quant in SUPPORTED_QUANTIZATIONS:
        if quant in name:
            return quant
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifica y carga modelos GGUF cuantizados (Q4_K_M, Q8_0)."
    )
    parser.add_argument("model_path", type=Path, help="Ruta al archivo .gguf")
    parser.add_argument(
        "--n-ctx", type=int, default=4096, help="Tamaño del contexto (default: 4096)"
    )
    parser.add_argument(
        "--load", action="store_true", help="Carga el modelo en memoria además de verificarlo"
    )
    args = parser.parse_args(argv)

    try:
        verify_gguf(args.model_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    quant = detect_quantization(args.model_path)
    if quant is None:
        print(
            f"[AVISO] No se pudo determinar la cuantización de {args.model_path.name}. "
            f"Soportadas: {', '.join(SUPPORTED_QUANTIZATIONS)}"
        )
    else:
        print(f"[OK] Cuantización detectada: {quant}")

    print(f"[OK] Archivo GGUF válido: {args.model_path}")

    if args.load:
        try:
            server = LLMServer(args.model_path, n_ctx=args.n_ctx)
        except ModelLoadError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"[OK] Modelo cargado correctamente (n_ctx={args.n_ctx}).")
        server.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
