#!/usr/bin/env python3
"""Reject environments that do not match the verified SERA release stack."""

from __future__ import annotations

import sys

import numpy
import torch
import torchvision


EXPECTED = {
    "python": (3, 10),
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "numpy": "2.2.6",
}


def main() -> None:
    actual = {
        "python": sys.version_info[:2],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
    }
    mismatches = [
        f"{name}: expected {EXPECTED[name]!r}, found {value!r}"
        for name, value in actual.items()
        if value != EXPECTED[name]
    ]
    if mismatches:
        details = "\n  ".join(mismatches)
        raise SystemExit(
            "SERA environment verification failed. Activate the environment "
            "created from environment.yml or set PYTHON_BIN explicitly.\n  "
            + details
        )
    print(
        "SERA environment verified: "
        f"Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}, "
        f"NumPy {numpy.__version__}."
    )


if __name__ == "__main__":
    main()
