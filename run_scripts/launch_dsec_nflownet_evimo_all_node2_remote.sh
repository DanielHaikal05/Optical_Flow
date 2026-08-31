#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

python_bin="${PYTHON_BIN:-/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"
group_root="EvMotionSeg/data/dsec_nflownet_evimo_all3_${timestamp}"
group_status="${group_root}/status.log"
build_dir="${NFLOWNET_EVMOTIONSEG_BUILD_DIR:-/tmp/dsec_evmotionseg_nflownet_evimo_build}"

downsample_rate="${EVMOTIONSEG_DOWNSAMPLE_RATE:-1}"
smooth_term="${EVMOTIONSEG_SMOOTH_TERM:-6000}"
label_term="${EVMOTIONSEG_LABEL_TERM:-60000}"
max_labels="${EVMOTIONSEG_MAX_LABELS:-24}"
preview_stride="${EVMOTIONSEG_PREVIEW_STRIDE:-4}"
imo_background_mode="${EVMOTIONSEG_IMO_BACKGROUND_MODE:-background_fit}"
flow_scale="${NFLOWNET_FLOW_SCALE:-1.0}"
min_gradient="${NFLOWNET_MIN_GRADIENT:-0.01}"
gradient_blur_sigma="${NFLOWNET_GRADIENT_BLUR_SIGMA:-1.0}"

if [[ -n "${DSEC_SEQUENCE_DIRS:-}" ]]; then
  read -r -a sequences <<<"${DSEC_SEQUENCE_DIRS}"
else
  sequences=(
    "Datasets/DSEC/zurich_city_00_a"
    "Datasets/DSEC/interlaken_00_c"
    "Datasets/DSEC/zurich_city_00_b"
  )
fi

mkdir -p "${group_root}"

log_group() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${group_status}"
}

log_group "Started DSEC NFlowNet-EVIMO all-events group on $(hostname)."
log_group "Group output: ${group_root}"
log_group "Sequences: ${sequences[*]}"
log_group "Checkpoint: model_weights.pth"
log_group "flow_scale=${flow_scale}, min_gradient=${min_gradient}, gradient_blur_sigma=${gradient_blur_sigma}"
log_group "EvMotionSeg terms: downsample=${downsample_rate}, smooth=${smooth_term}, label=${label_term}, max_labels=${max_labels}, imo_background_mode=${imo_background_mode}"

"${python_bin}" - <<'PY' >"${group_root}/dependency_check.log" 2>&1
import cv2
import h5py
import hdf5plugin
import numpy
import torch
import yaml
from PIL import Image
print("dependency check ok; cuda:", torch.cuda.is_available())
PY

for sequence_dir in "${sequences[@]}"; do
  sequence_label="$(basename "${sequence_dir}")"
  run_name="dsec_${sequence_label}_nflownet_evimo_all_down${downsample_rate}_bgfit_${timestamp}"
  ev_out="EvMotionSeg/data/${run_name}"
  mkdir -p "${ev_out}/logs"

  ln -sfn "$(basename "${ev_out}")" "EvMotionSeg/data/dsec_${sequence_label}_nflownet_evimo_all_latest"
  printf '%s\n' "${ev_out}" >"EvMotionSeg/data/dsec_${sequence_label}_nflownet_evimo_all_latest_path.txt"

  log_group "Starting ${sequence_label}; output=${ev_out}"
  "${python_bin}" tools/run_dsec_nflownet_evimo_all.py \
    --sequence-dir "${sequence_dir}" \
    --output-dir "${ev_out}" \
    --checkpoint model_weights.pth \
    --build-dir "${build_dir}" \
    --interval-s 0.1 \
    --preview-stride "${preview_stride}" \
    --flow-scale "${flow_scale}" \
    --min-gradient "${min_gradient}" \
    --gradient-blur-sigma "${gradient_blur_sigma}" \
    --downsample-rate "${downsample_rate}" \
    --smooth-term "${smooth_term}" \
    --label-term "${label_term}" \
    --max-labels "${max_labels}" \
    --imo-background-mode "${imo_background_mode}" \
    --overwrite \
    >"${ev_out}/logs/run_dsec_nflownet_evimo_all.log" 2>&1

  cp "${ev_out}/summary.json" "${group_root}/${sequence_label}_summary.json"
  log_group "Completed ${sequence_label}."
done

"${python_bin}" - "${group_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = [json.loads(path.read_text()) for path in sorted(root.glob("*_summary.json"))]
summary = {
    "group_root": str(root),
    "model": "NFlowNet downloaded EVIMO checkpoint",
    "checkpoint": "model_weights.pth",
    "sampling": "none",
    "sequences": items,
    "total_written_events": sum(item.get("written_events", 0) for item in items),
    "total_intervals": sum(item.get("num_intervals", 0) for item in items),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

log_group "DSEC NFlowNet-EVIMO all-events group complete."
