#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${1:?Usage: scripts/run_experiment.sh DATASET BACKBONE SHOTS [PROTOCOL] [SEED]}"
BACKBONE="${2:?Usage: scripts/run_experiment.sh DATASET BACKBONE SHOTS [PROTOCOL] [SEED]}"
SHOTS="${3:?Usage: scripts/run_experiment.sh DATASET BACKBONE SHOTS [PROTOCOL] [SEED]}"
PROTOCOL="${4:-full_val}"
SEED="${5:-2024}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/pretrained}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_UEC_ABLATION_JSON="${SAVE_UEC_ABLATION_JSON:-}"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 2; }
[[ -d "${MODEL_ROOT}/clip" ]] || { echo "Missing CLIP cache: ${MODEL_ROOT}/clip" >&2; exit 2; }

"${PYTHON_BIN}" scripts/check_environment.py

validation_mode="official_val"
routing_mode="official_val"
protocol_name="SERA_val"
extra_args=(
  --enable_residual_evidence_adaptation
  --enable_validation_protected_evidence_routing
  --evidence_routing_guard_margin 0.0
  --enable_multi_view_evidence_aggregation
  --residual_evidence_iters_per_shot 500
  --lora_position all
  --lora_encoder both
)

case "${PROTOCOL}" in
  full_val)
    ;;
  support_only)
    validation_mode="support"
    routing_mode="support_only"
    protocol_name="SERA_support_only"
    ;;
  support_fixed)
    validation_mode="support"
    routing_mode="fixed"
    protocol_name="SERA_support_fixed"
    ;;
  no_rea)
    protocol_name="SERA_no_REA"
    extra_args=(
      --enable_validation_protected_evidence_routing
      --evidence_routing_guard_margin 0.0
      --enable_multi_view_evidence_aggregation
      --lora_position all
      --lora_encoder both
    )
    ;;
  no_mea)
    protocol_name="SERA_no_MEA"
    extra_args=(
      --enable_residual_evidence_adaptation
      --enable_validation_protected_evidence_routing
      --evidence_routing_guard_margin 0.0
      --residual_evidence_iters_per_shot 500
      --lora_position all
      --lora_encoder both
    )
    ;;
  *)
    echo "Unknown protocol: ${PROTOCOL}" >&2
    exit 2
    ;;
esac

public_dataset="${DATASET}"
[[ "${DATASET}" == "fgvc" ]] && public_dataset="fgvc_aircraft"
run_dir="${OUTPUT_ROOT}/${PROTOCOL}"
mkdir -p "${run_dir}"
stem="${protocol_name}_${public_dataset}_${BACKBONE}_${SHOTS}shot_seed${SEED}"
uec_ablation_args=()
if [[ -n "${SAVE_UEC_ABLATION_JSON}" ]]; then
  mkdir -p "$(dirname "${SAVE_UEC_ABLATION_JSON}")"
  uec_ablation_args=(--save_uec_ablation_json "${SAVE_UEC_ABLATION_JSON}")
fi

"${PYTHON_BIN}" train.py \
  --dataset "${DATASET}" \
  --root_path "${DATA_ROOT}" \
  --model_cache_dir "${MODEL_ROOT}/clip" \
  --backbone "${BACKBONE}" \
  --shots "${SHOTS}" \
  --seed "${SEED}" \
  --epochs 50 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --validation_mode "${validation_mode}" \
  --evidence_pool_profile balanced \
  --evidence_routing_mode "${routing_mode}" \
  "${extra_args[@]}" \
  "${uec_ablation_args[@]}" \
  --save_json "${run_dir}/${stem}.json"
