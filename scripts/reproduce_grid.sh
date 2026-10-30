#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL="${1:-full_val}"

datasets=(caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101 imagenet)
backbones=(vit_b16 vit_b32 vit_l14)
shots=(1 2 4 8 16)

for backbone in "${backbones[@]}"; do
  for shot in "${shots[@]}"; do
    for dataset in "${datasets[@]}"; do
      "${ROOT_DIR}/scripts/run_experiment.sh" "${dataset}" "${backbone}" "${shot}" "${PROTOCOL}" 2024
    done
  done
done

