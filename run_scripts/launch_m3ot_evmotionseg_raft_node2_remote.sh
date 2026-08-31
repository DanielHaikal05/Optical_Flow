#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

sequence_dir="${M3OT_SEQUENCE_DIR:-Datasets/M3OT/1/rgb/test/1-03}"
sequence_label="${M3OT_SEQUENCE_LABEL:-1_rgb_test_1-03}"
num_intervals="${RAFT_EVMOTIONSEG_NUM_INTERVALS:-80}"
start_frame="${RAFT_EVMOTIONSEG_START_FRAME:-1}"
max_points_per_frame="${RAFT_EVMOTIONSEG_MAX_POINTS_PER_FRAME:-20000}"
random_fraction="${RAFT_EVMOTIONSEG_RANDOM_FRACTION:-0.25}"
downsample_rate="${EVMOTIONSEG_DOWNSAMPLE_RATE:-1}"
raft_model="${RAFT_EVMOTIONSEG_MODEL:-small}"
raft_updates="${RAFT_EVMOTIONSEG_UPDATES:-12}"
run_prefix="${EVMOTIONSEG_RUN_PREFIX:-m3ot_${sequence_label}_raft_${raft_model}_${max_points_per_frame}_down${downsample_rate}_bgfit}"
run_name="${run_prefix}_$(date +%Y%m%d_%H%M%S)"
ev_out="EvMotionSeg/data/${run_name}"
latest_link="EvMotionSeg/data/${run_prefix}_latest"
latest_path_file="EvMotionSeg/data/${run_prefix}_latest_path.txt"
status_log="${ev_out}/logs/status.log"
python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
build_dir="/tmp/m3ot_evmotionseg_raft_build"
interval_s="0.1"
width="640"
height="512"
fx="640"
fy="640"

mkdir -p "${ev_out}/logs"
ln -sfn "$(basename "${ev_out}")" "${latest_link}"
printf '%s\n' "${ev_out}" > "${latest_path_file}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Started M3OT RAFT -> EvMotionSeg run on $(hostname)."
log "Output directory: ${ev_out}"
log "Sequence: ${sequence_dir}"
log "Frames: start=${start_frame}, intervals=${num_intervals}, interval=${interval_s}s"
log "Sampling: max ${max_points_per_frame} pseudo-events/frame; random_fraction=${random_fraction}; downsample_rate=${downsample_rate}"
log "RAFT: model=${raft_model}, updates=${raft_updates}"
log "EvMotionSeg terms: data=1, smooth=6000, label=60000, max_labels=24"

log "Checking Python dependencies."
"${python_bin}" - <<'PY' >"${ev_out}/logs/dependency_check.log" 2>&1
import cv2
import numpy
import torch
import torchvision
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available())
print("dependency check ok")
PY

log "Building EvMotionSeg portable standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" \
  >"${ev_out}/logs/build_standalone.log" 2>&1

log "Preparing EvMotionSeg text inputs from M3OT frames and RAFT flow."
"${python_bin}" EvMotionSeg/tools/prepare_m3ot_raft_for_evmotionseg.py \
  --sequence-dir "${sequence_dir}" \
  --output-dir "${ev_out}" \
  --start-frame "${start_frame}" \
  --num-intervals "${num_intervals}" \
  --max-points-per-frame "${max_points_per_frame}" \
  --random-fraction "${random_fraction}" \
  --preview-stride 4 \
  --raft-model "${raft_model}" \
  --raft-updates "${raft_updates}" \
  >"${ev_out}/logs/prepare_raft.log" 2>&1

prepared_intervals="$("${python_bin}" - "${ev_out}/evmotionseg_input_summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["num_intervals"])
PY
)"

log "Prepared ${prepared_intervals} intervals. Starting EvMotionSeg."
"${build_dir}/motion_segmentation_standalone" \
  --data_file_path "${ev_out}" \
  --interval "${interval_s}" \
  --width "${width}" \
  --height "${height}" \
  --downsample_rate "${downsample_rate}" \
  --fx "${fx}" \
  --fy "${fy}" \
  --data_term 1 \
  --smooth_term 6000 \
  --label_term 60000 \
  --GraphCutIteration 10 \
  --MotionSegIteration 4 \
  --max_labels 24 \
  --num_intervals "${prepared_intervals}" \
  --imo_background_mode background_fit \
  >"${ev_out}/logs/evmotionseg.log" 2>&1

log "Generating qualitative overlay contact sheet, video, and summary."
"${python_bin}" EvMotionSeg/tools/summarize_evmotionseg_run.py "${ev_out}" \
  --label "m3ot_${sequence_label}_raft_${raft_model}_${max_points_per_frame}" \
  --stride 4 \
  >"${ev_out}/logs/qualitative.log" 2>&1

log "M3OT RAFT -> EvMotionSeg run complete."
