#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_SOURCE="${1:?Usage: scripts/link_resources.sh DATASET_ROOT [MODEL_ROOT]}"
MODEL_SOURCE="${2:-}"

DATA_SOURCE="$(realpath "${DATA_SOURCE}")"
[[ -d "${DATA_SOURCE}" ]] || { echo "Dataset root does not exist: ${DATA_SOURCE}" >&2; exit 2; }

if [[ -e "${ROOT_DIR}/data" && ! -L "${ROOT_DIR}/data" ]]; then
  echo "Refusing to replace the package-local data directory: ${ROOT_DIR}/data" >&2
  echo "A full reproducibility bundle does not require resource linking." >&2
  exit 3
fi

ln -sfn "${DATA_SOURCE}" "${ROOT_DIR}/data"

if [[ -n "${MODEL_SOURCE}" ]]; then
  MODEL_SOURCE="$(realpath "${MODEL_SOURCE}")"
  [[ -d "${MODEL_SOURCE}" ]] || { echo "Model root does not exist: ${MODEL_SOURCE}" >&2; exit 2; }
  [[ ! -L "${ROOT_DIR}/pretrained" ]] || {
    echo "Refusing to copy into a symlinked pretrained directory; replace it with a local directory first." >&2
    exit 3
  }
  mkdir -p "${ROOT_DIR}/pretrained"
  cp -a "${MODEL_SOURCE}/." "${ROOT_DIR}/pretrained/"
fi

[[ -d "${ROOT_DIR}/pretrained/clip" ]] || {
  echo "Missing local pretrained/clip cache; pass MODEL_ROOT once to copy it into the package." >&2
  exit 2
}

echo "data -> $(readlink -f "${ROOT_DIR}/data")"
echo "pretrained (package-local) -> $(realpath "${ROOT_DIR}/pretrained")"
