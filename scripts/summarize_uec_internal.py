from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("SERA_UEC_internal_*.json")):
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
                    "source_json": str(path),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset"]
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    group_specs = (
        ("backbone_shot", lambda row: (row["variant"], row["backbone"], row["shots"])),
        ("backbone", lambda row: (row["variant"], row["backbone"], "all")),
        ("overall", lambda row: (row["variant"], "all", "all")),
    )
    for level, key_fn in group_specs:
        groups = defaultdict(list)
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
                    "mean_full_acc": sum(row["full_final_acc"] for row in members) / len(members),
                    "mean_ablation_acc": sum(row["ablation_final_acc"] for row in members) / len(members),
                    "mean_delta_vs_full_pp": sum(row["delta_vs_full_pp"] for row in members) / len(members),
                }
            )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]) if summary_rows else ["level"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({
        "paired_json": len(list(args.root.glob('SERA_UEC_internal_*.json'))),
        "result_rows": len(rows),
        "summary_rows": len(summary_rows),
        "out": str(args.out),
        "summary_out": str(args.summary_out),
    }, indent=2))


if __name__ == "__main__":
    main()
