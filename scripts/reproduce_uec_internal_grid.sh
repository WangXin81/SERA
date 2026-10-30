#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/uec_internal_ablation/full_outputs}"
UEC_OUTPUT_ROOT="${UEC_OUTPUT_ROOT:-${ROOT_DIR}/outputs/uec_internal_ablation/paired_uec_outputs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2024}"

datasets=(caltech101 dtd eurosat fgvc food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101)
backbones=(vit_b32 vit_b16 vit_l14)
shots_list=(1 2 4 8 16)

mkdir -p "${OUTPUT_ROOT}" "${UEC_OUTPUT_ROOT}"
manifest="${UEC_OUTPUT_ROOT}/grid_manifest.tsv"
printf "dataset\tbackbone\tshots\tseed\tfull_json\tpaired_uec_json\n" > "${manifest}"

for dataset in "${datasets[@]}"; do
  public_dataset="${dataset}"
  [[ "${dataset}" == "fgvc" ]] && public_dataset="fgvc_aircraft"
  for backbone in "${backbones[@]}"; do
    for shots in "${shots_list[@]}"; do
      stem="SERA_val_${public_dataset}_${backbone}_${shots}shot_seed${SEED}"
      full_json="${OUTPUT_ROOT}/full_val/${stem}.json"
      paired_json="${UEC_OUTPUT_ROOT}/SERA_UEC_internal_${public_dataset}_${backbone}_${shots}shot_seed${SEED}.json"
      if [[ -e "${full_json}" || -e "${paired_json}" ]]; then
        echo "Refusing to overwrite an existing grid cell: ${dataset} ${backbone} ${shots}" >&2
        exit 3
      fi
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${dataset}" "${backbone}" "${shots}" "${SEED}" "${full_json}" "${paired_json}" >> "${manifest}"
      printf "[UEC_CELL_START] %s %s %s %s\n" "$(date -Iseconds)" "${dataset}" "${backbone}" "${shots}"
      SAVE_UEC_ABLATION_JSON="${paired_json}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      NUM_WORKERS="${NUM_WORKERS}" \
        bash scripts/run_experiment.sh "${dataset}" "${backbone}" "${shots}" full_val "${SEED}"
      printf "[UEC_CELL_DONE] %s %s %s %s\n" "$(date -Iseconds)" "${dataset}" "${backbone}" "${shots}"
    done
  done
done

"${PYTHON_BIN}" scripts/summarize_uec_internal.py \
  "${UEC_OUTPUT_ROOT}" \
  --out "${UEC_OUTPUT_ROOT}/UEC_internal_ablation_330_detail.csv" \
  --summary-out "${UEC_OUTPUT_ROOT}/UEC_internal_ablation_summary.csv"

printf "[UEC_GRID_DONE] %s\n" "$(date -Iseconds)"
