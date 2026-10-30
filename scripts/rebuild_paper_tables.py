#!/usr/bin/env python3
"""Rebuild and verify all paper-facing result tables from frozen JSON records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_FIELDS = [
    "status",
    "variant",
    "backbone",
    "shots",
    "dataset",
    "acc",
    "query_leakage",
    "test_label_used",
    "selection_mode",
    "validation_mode",
    "official_val_used_for_selection",
    "support_only_selection_used",
    "fixed_selection_used",
    "file",
    "source_dir",
    "error",
]
CORE_VARIANTS = (
    ("SERA_val", "full", None),
    ("SERA_no_REA", "no_rea", None),
    ("SERA_no_MEA", "no_mea", None),
    ("SERA_support_only", "routing_protocols", "support_only"),
    ("SERA_support_fixed", "routing_protocols", "fixed"),
)
EXPECTED_CORE_PER_VARIANT = 165
EXPECTED_UEC_PAIRED = 165
DATASET_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "caltech101",
            "dtd",
            "eurosat",
            "fgvc_aircraft",
            "food101",
            "oxford_flowers",
            "oxford_pets",
            "stanford_cars",
            "sun397",
            "ucf101",
            "imagenet",
        )
    )
}
BACKBONE_ORDER = {name: index for index, name in enumerate(("vit_b16", "vit_b32", "vit_l14"))}
SHOT_ORDER = {value: index for index, value in enumerate((16, 1, 2, 4, 8))}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_core_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    coverage: list[dict] = []
    for variant, source_dir, selection_mode in CORE_VARIANTS:
        variant_rows = []
        root = ROOT / "results" / "reference" / source_dir
        for path in sorted(root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if selection_mode and payload.get("selection_mode") != selection_mode:
                continue
            metric_means = payload.get("metric_means", {})
            final = metric_means.get("final", metric_means.get("final_acc"))
            if not isinstance(final, (int, float)):
                raise ValueError(f"{path.name}: missing final accuracy")
            variant_rows.append(
                {
                    "status": "OK",
                    "variant": variant,
                    "backbone": payload["backbone"],
                    "shots": payload["shots"],
                    "dataset": payload["dataset"],
                    "acc": final * 100.0,
                    "query_leakage": payload["query_leakage"],
                    "test_label_used": payload["test_label_used"],
                    "selection_mode": payload["selection_mode"],
                    "validation_mode": payload["validation_mode"],
                    "official_val_used_for_selection": payload[
                        "official_val_used_for_selection"
                    ],
                    "support_only_selection_used": payload[
                        "support_only_selection_used"
                    ],
                    "fixed_selection_used": payload["fixed_selection_used"],
                    "file": path.relative_to(ROOT).as_posix(),
                    "source_dir": source_dir,
                    "error": "",
                }
            )
        if len(variant_rows) != EXPECTED_CORE_PER_VARIANT:
            raise ValueError(
                f"{variant}: expected {EXPECTED_CORE_PER_VARIANT} rows, "
                f"found {len(variant_rows)}"
            )
        variant_rows.sort(
            key=lambda row: (
                DATASET_ORDER[row["dataset"]],
                BACKBONE_ORDER[row["backbone"]],
                SHOT_ORDER[row["shots"]],
            )
        )
        rows.extend(variant_rows)
        coverage.append(
            {"check": f"{variant}_count", "status": "OK", "detail": len(variant_rows)}
        )
    return rows, coverage


def build_uec_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    paired_root = ROOT / "results" / "reference" / "uec_internal" / "paired_json"
    paths = sorted(paired_root.glob("SERA_UEC_internal_*.json"))
    if len(paths) != EXPECTED_UEC_PAIRED:
        raise ValueError(
            f"UEC: expected {EXPECTED_UEC_PAIRED} paired JSON files, found {len(paths)}"
        )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for mode, variant in payload["variants"].items():
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "backbone": payload["backbone"],
                    "shots": payload["shots"],
                    "seed": payload["seed"],
                    "variant": mode,
                    "removed_from_routing_pool": variant["removed_from_routing_pool"],
                    "full_final_acc": payload["full_final_acc"],
                    "ablation_final_acc": variant["final_acc"],
                    "delta_vs_full_pp": variant["delta_vs_full_pp"],
                    "selection_key": variant["selection_key"],
                    "guard_decision": variant["guard_decision"],
                    "base_branch_count": variant["base_branch_count"],
                    "candidate_branch_count": variant["candidate_branch_count"],
                    "query_leakage": payload["query_leakage"],
                    "test_label_used": payload["test_label_used"],
                    "source_json": path.relative_to(ROOT).as_posix(),
                }
            )

    summary_rows: list[dict] = []
    group_specs = (
        ("backbone_shot", lambda row: (row["variant"], row["backbone"], row["shots"])),
        ("backbone", lambda row: (row["variant"], row["backbone"], "all")),
        ("overall", lambda row: (row["variant"], "all", "all")),
    )
    for level, key_fn in group_specs:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        for (variant, backbone, shots), members in sorted(groups.items()):
            summary_rows.append(
                {
                    "level": level,
                    "variant": variant,
                    "backbone": backbone,
                    "shots": shots,
                    "cells": len(members),
                    "mean_full_acc": sum(row["full_final_acc"] for row in members)
                    / len(members),
                    "mean_ablation_acc": sum(
                        row["ablation_final_acc"] for row in members
                    )
                    / len(members),
                    "mean_delta_vs_full_pp": sum(
                        row["delta_vs_full_pp"] for row in members
                    )
                    / len(members),
                }
            )
    return rows, summary_rows


def assert_same(generated: Path, archived: Path) -> None:
    generated_rows = read_csv(generated)
    archived_rows = read_csv(archived)
    if generated_rows != archived_rows:
        for index, (new, old) in enumerate(zip(generated_rows, archived_rows), start=2):
            equivalent = new.keys() == old.keys()
            if equivalent:
                for field in new:
                    if new[field] == old[field]:
                        continue
                    try:
                        equivalent = math.isclose(
                            float(new[field]), float(old[field]), rel_tol=0.0, abs_tol=1e-12
                        )
                    except ValueError:
                        equivalent = False
                    if not equivalent:
                        break
            if not equivalent:
                raise SystemExit(
                    f"Table mismatch: {archived.relative_to(ROOT)} row {index}\n"
                    f"generated={new}\narchived={old}"
                )
        if len(generated_rows) != len(archived_rows):
            raise SystemExit(
                f"Table length mismatch: {archived.relative_to(ROOT)} "
                f"generated={len(generated_rows)} archived={len(archived_rows)}"
            )


def rebuild(output_dir: Path) -> None:
    core_rows, coverage_rows = build_core_rows()
    uec_rows, uec_summary = build_uec_rows()

    core_path = output_dir / "sera_common11_results.csv"
    coverage_path = output_dir / "sera_common11_coverage.csv"
    uec_path = output_dir / "uec_330_detail.csv"
    uec_summary_path = output_dir / "uec_summary.csv"
    write_csv(core_path, core_rows, CORE_FIELDS)
    write_csv(coverage_path, coverage_rows, ["check", "status", "detail"])
    write_csv(uec_path, uec_rows, list(uec_rows[0]))
    write_csv(uec_summary_path, uec_summary, list(uec_summary[0]))

    archived_uec = ROOT / "results" / "reference" / "uec_internal" / "tables"
    assert_same(core_path, ROOT / "results" / "tables" / core_path.name)
    assert_same(coverage_path, ROOT / "results" / "tables" / coverage_path.name)
    assert_same(uec_path, archived_uec / "uec_330_detail.csv")
    assert_same(uec_summary_path, archived_uec / "uec_summary.csv")
    print(
        "Paper-table reconstruction PASS: "
        f"{len(core_rows)} core rows, {len(uec_rows)} UEC ablation rows, "
        f"{len(uec_summary)} UEC summary rows."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Retain regenerated tables here; default uses a temporary directory.",
    )
    args = parser.parse_args()
    if args.output_dir:
        rebuild(args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="sera_tables_") as directory:
            rebuild(Path(directory))


if __name__ == "__main__":
    main()

