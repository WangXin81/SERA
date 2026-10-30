#!/usr/bin/env python3
"""Verify package structure, reference coverage, and leakage declarations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "full": 165,
    "no_rea": 165,
    "no_mea": 165,
    "routing_protocols": 330,
}
UEC_PAIRED_COUNT = 165
UEC_VARIANTS = {
    "base_alignment_only": "vocabulary_free_relation_evidence",
    "vocabulary_free_relation_only": "base_alignment_evidence",
}
EXPECTED_DATASETS = {
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "imagenet",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
}
EXPECTED_BACKBONES = {"vit_b16", "vit_b32", "vit_l14"}
EXPECTED_SHOTS = {1, 2, 4, 8, 16}
FORBIDDEN_PUBLIC_MARKERS = {
    "ape_feat_nums",
    "clip_lora",
    "ldc_beta",
    "prolip",
    "taskres",
    "tipf_",
    "legacy_",
    "_compat",
    "sera_final_v1",
    "results_sera_final_v1",
    "recent_sota",
    "/root/autodl",
    "connect.westc.seetacloud.com",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Verify the source distribution without requiring local model weights.",
    )
    verification_args = parser.parse_args()
    required = [
        ROOT / "train.py",
        ROOT / "sera" / "runner.py",
        ROOT / "sera" / "clip_encoder" / "__init__.py",
        ROOT / "datasets" / "__init__.py",
        ROOT / "sera" / "low_rank" / "__init__.py",
        ROOT / "scripts" / "check_environment.py",
        ROOT / "scripts" / "run_experiment.sh",
        ROOT / "scripts" / "reproduce_grid.sh",
        ROOT / "scripts" / "reproduce_uec_internal_grid.sh",
        ROOT / "scripts" / "summarize_uec_internal.py",
        ROOT / "scripts" / "rebuild_paper_tables.py",
        ROOT / "README_SUPPLEMENTARY.txt",
        ROOT / "docs" / "VERIFICATION.md",
        ROOT / "results" / "tables" / "sera_common11_results.csv",
        ROOT / "results" / "reference" / "uec_internal" / "README.md",
        ROOT / "results" / "reference" / "uec_internal" / "tables" / "uec_330_detail.csv",
        ROOT / "results" / "reference" / "uec_internal" / "tables" / "uec_summary.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))

    failures: list[str] = []

    if not verification_args.source_only:
        pretrained = ROOT / "pretrained"
        if pretrained.is_symlink():
            failures.append("pretrained must be package-local, not a symlink")
        for model_name in ("ViT-B-32.pt", "ViT-B-16.pt", "ViT-L-14.pt"):
            if not (pretrained / "clip" / model_name).is_file():
                failures.append(f"Missing package-local pretrained/clip/{model_name}")

    runner_text = (ROOT / "sera" / "runner.py").read_text(encoding="utf-8")
    required_runner_fragments = {
        'parser.add_argument("--batch_size", default=32': "paper batch-size default",
        'parser.add_argument("--lora_position", default="all"': "paper LoRA-position default",
        '"--save_uec_ablation_json"': "paired UEC ablation entry point",
        '"vocabulary_free_relation_only"': "SERA w/o BA evidence-family variant",
    }
    for fragment, label in required_runner_fragments.items():
        if fragment not in runner_text:
            failures.append(f"Runner is missing {label}")

    dataset_utils = (ROOT / "datasets" / "utils.py").read_text(encoding="utf-8")
    if 'Path(__file__).resolve().parent / "jsons" / resolved.name' not in dataset_utils:
        failures.append("Dataset loaders do not fall back to bundled split metadata")

    public_roots = [
        ROOT / "README.md",
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "results" / "reference",
        ROOT / "scripts",
        ROOT / "sera",
        ROOT / "datasets",
        ROOT / "train.py",
    ]
    for public_root in public_roots:
        paths = [public_root] if public_root.is_file() else public_root.rglob("*")
        for path in paths:
            if (
                not path.is_file()
                or path.resolve() == Path(__file__).resolve()
                or path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg"}
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for marker in FORBIDDEN_PUBLIC_MARKERS:
                if marker in text:
                    failures.append(
                        f"Public-name audit: {path.relative_to(ROOT)} contains {marker!r}"
                    )

    for group, expected in EXPECTED_COUNTS.items():
        paths = sorted((ROOT / "results" / "reference" / group).glob("*.json"))
        if len(paths) != expected:
            failures.append(f"{group}: expected {expected} JSON files, found {len(paths)}")
        observed: set[tuple[str, str, int, str]] = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            args = payload.get("args", {})
            if args.get("root_path") != "./data":
                failures.append(f"{path.name}: non-public dataset root metadata")
            if args.get("model_cache_dir") != "./pretrained/clip":
                failures.append(f"{path.name}: non-public model root metadata")
            if "selected_feature_counts" not in args:
                failures.append(f"{path.name}: missing public selected-feature metadata")
            if group == "no_rea":
                if args.get("enable_residual_evidence") is not False:
                    failures.append(f"{path.name}: REA is not disabled")
                if args.get("enable_residual_reference_warmup") is not False:
                    failures.append(f"{path.name}: residual-reference warmup is not disabled")
                if args.get("lora_position") not in {"all", "top3"}:
                    failures.append(f"{path.name}: unexpected inactive lora_position metadata")
            elif args.get("lora_position") != "all":
                failures.append(f"{path.name}: lora_position is not 'all'")
            if payload.get("query_leakage") is not False:
                failures.append(f"{path.name}: query_leakage is not false")
            if payload.get("test_label_used") is not False:
                failures.append(f"{path.name}: test_label_used is not false")
            observed.add(
                (
                    payload.get("dataset"),
                    payload.get("backbone"),
                    payload.get("shots"),
                    payload.get("selection_mode"),
                )
            )

        expected_modes = {"official_val"}
        if group == "routing_protocols":
            expected_modes = {"support_only", "fixed"}
        expected_grid = {
            (dataset, backbone, shots, mode)
            for dataset in EXPECTED_DATASETS
            for backbone in EXPECTED_BACKBONES
            for shots in EXPECTED_SHOTS
            for mode in expected_modes
        }
        missing_cells = expected_grid - observed
        extra_cells = observed - expected_grid
        if missing_cells:
            failures.append(f"{group}: missing {len(missing_cells)} grid cells")
        if extra_cells:
            failures.append(f"{group}: found {len(extra_cells)} unexpected grid cells")

    uec_root = ROOT / "results" / "reference" / "uec_internal"
    uec_paths = sorted((uec_root / "paired_json").glob("*.json"))
    if len(uec_paths) != UEC_PAIRED_COUNT:
        failures.append(
            f"uec_internal: expected {UEC_PAIRED_COUNT} paired JSON files, found {len(uec_paths)}"
        )
    uec_observed: set[tuple[str, str, int]] = set()
    for path in uec_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_mode") != "official_val":
            failures.append(f"{path.name}: UEC selection_mode is not official_val")
        if payload.get("seed") != 2024:
            failures.append(f"{path.name}: UEC seed is not 2024")
        if payload.get("same_run_shared_training_and_logits") is not True:
            failures.append(f"{path.name}: UEC paired-run declaration is not true")
        if payload.get("query_leakage") is not False:
            failures.append(f"{path.name}: UEC query_leakage is not false")
        if payload.get("test_label_used") is not False:
            failures.append(f"{path.name}: UEC test_label_used is not false")
        variants = payload.get("variants", {})
        if set(variants) != set(UEC_VARIANTS):
            failures.append(f"{path.name}: unexpected UEC variant set")
        for variant_name, removed_family in UEC_VARIANTS.items():
            variant = variants.get(variant_name, {})
            if not isinstance(variant.get("final_acc"), (int, float)):
                failures.append(f"{path.name}: {variant_name} missing final_acc")
            if variant.get("removed_from_routing_pool") != removed_family:
                failures.append(f"{path.name}: {variant_name} removes the wrong evidence family")
            if variant.get("shared_feature_training_unchanged") is not True:
                failures.append(f"{path.name}: {variant_name} changed shared feature training")
        uec_observed.add(
            (payload.get("dataset"), payload.get("backbone"), payload.get("shots"))
        )

    expected_uec_grid = {
        (dataset, backbone, shots)
        for dataset in EXPECTED_DATASETS
        for backbone in EXPECTED_BACKBONES
        for shots in EXPECTED_SHOTS
    }
    if expected_uec_grid - uec_observed:
        failures.append(
            f"uec_internal: missing {len(expected_uec_grid - uec_observed)} grid cells"
        )
    if uec_observed - expected_uec_grid:
        failures.append(
            f"uec_internal: found {len(uec_observed - expected_uec_grid)} unexpected grid cells"
        )

    detail_path = uec_root / "tables" / "uec_330_detail.csv"
    with detail_path.open("r", encoding="utf-8-sig", newline="") as handle:
        detail_rows = list(csv.DictReader(handle))
    if len(detail_rows) != 330:
        failures.append(f"uec_330_detail.csv: expected 330 rows, found {len(detail_rows)}")
    variant_counts = {name: 0 for name in UEC_VARIANTS}
    for row in detail_rows:
        variant = row.get("variant")
        if variant in variant_counts:
            variant_counts[variant] += 1
        source_json = row.get("source_json", "")
        expected_prefix = "results/reference/uec_internal/paired_json/"
        if not source_json.startswith(expected_prefix):
            failures.append("uec_330_detail.csv: source_json is not package-relative")
            break
        if not (ROOT / source_json).is_file():
            failures.append(f"uec_330_detail.csv: missing source_json {source_json}")
            break
        if row.get("query_leakage") != "False" or row.get("test_label_used") != "False":
            failures.append("uec_330_detail.csv: leakage declaration is not false")
            break
    for variant, count in variant_counts.items():
        if count != 165:
            failures.append(f"uec_330_detail.csv: {variant} expected 165 rows, found {count}")

    if failures:
        raise SystemExit("Verification failed:\n" + "\n".join(failures))

    print("Package verification passed.")
    if verification_args.source_only:
        print("  source-only mode: local model-weight checks skipped")
    for group, count in EXPECTED_COUNTS.items():
        print(f"  {group}: {count} standalone reference JSON files")
    print(f"  uec_internal: {UEC_PAIRED_COUNT} paired JSON files / 330 ablation results")
    print("  total: 990 archived JSON files / 1155 paper-facing results")


if __name__ == "__main__":
    main()
