#!/usr/bin/env python3
"""Validate the eleven-dataset image tree without running model inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sera.runner import build_dataset


DATASETS = (
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
    "imagenet",
)

# Expected cache-free full-bundle inventories: (file count, total bytes).
DATA_INVENTORY = {
    "Caltech101": (9148, 139847023),
    "dtd": (5677, 634807823),
    "eurosat": (27001, 94850852),
    "fgvc_aircraft": (10021, 2763197764),
    "Food101": (101011, 5131457658),
    "Flower102": (8194, 347978184),
    "OxfordPets": (18476, 820145087),
    "StanfordCars": (16186, 2201742041),
    "sun397": (39701, 14564879711),
    "UCF101": (13323, 142676787),
    "imagenet": (66002, 8545877820),
}


def inventory_tree(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.lower().startswith("cache")]
        for filename in filenames:
            path = Path(directory) / filename
            count += 1
            size += path.stat().st_size
    return count, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--shots", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    for directory, expected in DATA_INVENTORY.items():
        actual = inventory_tree(args.root / directory)
        if actual != expected:
            failures.append(
                f"{directory}: inventory {actual}, expected {expected} "
                "(file count, total bytes)"
            )
        print(
            f"{directory:15s} inventory_files={actual[0]} "
            f"inventory_bytes={actual[1]} expected={actual == expected}"
        )

    for name in DATASETS:
        dataset_root, dataset = build_dataset(name, str(args.root), args.shots)
        split_sizes: dict[str, int] = {}
        missing: list[str] = []

        for split_name, items in (
            ("train", dataset.train_x),
            ("val", dataset.val),
            ("test", dataset.test),
        ):
            split_sizes[split_name] = len(items)
            for item in items:
                if not Path(item.impath).is_file():
                    missing.append(item.impath)
                    if len(missing) == 5:
                        break
            if missing:
                break

        if missing:
            failures.append(f"{name}: missing files: {missing}")

        print(
            f"{name:15s} root={dataset_root} "
            f"train={split_sizes.get('train', 0)} "
            f"val={split_sizes.get('val', 0)} "
            f"test={split_sizes.get('test', 0)} "
            f"missing={len(missing)}"
        )

    if failures:
        raise SystemExit("Dataset verification failed:\n" + "\n".join(failures))

    print("Dataset verification passed: all eleven datasets are complete.")


if __name__ == "__main__":
    main()
