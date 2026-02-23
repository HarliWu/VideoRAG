#!/usr/bin/env python3
"""Validate VideoRAG dependency compatibility for the CUDA 13.0 setup."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception as exc:  # pragma: no cover
    print(f"[ERROR] Missing 'packaging' dependency: {exc}")
    sys.exit(2)


REQUIRED_SPECS = {
    "numpy": "==1.26.4",
    "torch": "==2.9.1",
    "torchvision": "==0.24.1",
    "torchaudio": "==2.9.1",
    "accelerate": ">=0.30.1",
    "bitsandbytes": "==0.49.0",
    "moviepy": "==1.0.3",
    "ctranslate2": ">=4.7.1,<5",
    "faster-whisper": ">=1.2.1,<2",
    "transformers": ">=4.57.0,<4.58.0",
    "tokenizers": ">=0.22.0,<0.24.0",
    "sentencepiece": ">=0.2.0",
    "tiktoken": ">=0.9.0,<1",
    "openai": ">=1,<2",
    "tenacity": ">=8.2,<10",
    "ollama": "==0.5.3",
}


def read_version(pkg_name: str) -> str | None:
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None


def check_spec(pkg_name: str, spec: str, failures: list[str], warnings: list[str]) -> None:
    installed = read_version(pkg_name)
    if installed is None:
        failures.append(f"{pkg_name}: not installed (required {spec})")
        return
    if Version(installed) not in SpecifierSet(spec):
        failures.append(f"{pkg_name}: installed {installed}, expected {spec}")
        return
    warnings.append(f"{pkg_name}: {installed} (ok)")


def check_torch_family(failures: list[str], warnings: list[str]) -> None:
    torch_v = read_version("torch")
    tv_v = read_version("torchvision")
    ta_v = read_version("torchaudio")
    if not torch_v or not tv_v or not ta_v:
        return

    if not (torch_v.split(".")[:2] == tv_v.split(".")[:2] == ta_v.split(".")[:2]):
        failures.append(
            f"torch family mismatch: torch={torch_v}, torchvision={tv_v}, torchaudio={ta_v}"
        )
    else:
        warnings.append(
            f"torch family aligned: torch={torch_v}, torchvision={tv_v}, torchaudio={ta_v}"
        )


def check_cuda_runtime(warnings: list[str]) -> None:
    try:
        import torch
    except Exception:
        return

    cuda_version = torch.version.cuda
    if not cuda_version:
        warnings.append("torch CUDA runtime: CPU-only build detected")
        return

    warnings.append(f"torch CUDA runtime: {cuda_version}")
    if cuda_version.startswith("13."):
        # CTranslate2 prebuilt wheels currently document CUDA 12.x GPU support.
        # VideoRAG ASR code falls back to CPU when GPU ASR init fails.
        if platform.system() in {"Linux", "Windows"}:
            warnings.append(
                "note: ctranslate2 prebuilt wheels target CUDA 12.x for GPU. "
                "On CUDA 13.x, faster-whisper may run on CPU fallback."
            )


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for pkg_name, spec in REQUIRED_SPECS.items():
        check_spec(pkg_name, spec, failures, warnings)

    check_torch_family(failures, warnings)
    check_cuda_runtime(warnings)

    print("Dependency compatibility report")
    print("=" * 32)
    for msg in warnings:
        print(f"[OK] {msg}")
    if failures:
        for msg in failures:
            print(f"[FAIL] {msg}")
        return 1

    print("[PASS] All required package constraints are satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

