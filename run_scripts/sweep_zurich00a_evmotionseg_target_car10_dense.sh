#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

export PYTHONPATH="/home/DanielH/Optical_Flow/VecKM_flow:${PYTHONPATH:-}"

python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
sequence_label="${DSEC_SEQUENCE_LABEL:-$(basename "${sequence_dir}")}"
window_start="${ZURICH00A_DENSE_WINDOW_START:-144}"
window_intervals="${ZURICH00A_DENSE_WINDOW_INTERVALS:-16}"
target_frames="${ZURICH00A_TARGET_FRAMES:-144,145,146,147,148,155,156,157,158,159}"
max_events_per_interval="${ZURICH00A_DENSE_MAX_EVENTS_PER_INTERVAL:-50000}"
density_power="${ZURICH00A_DENSE_DENSITY_POWER:-0.75}"
density_offset="${ZURICH00A_DENSE_DENSITY_OFFSET:-1.0}"
grid_cols="${ZURICH00A_DENSE_GRID_COLS:-8}"
grid_rows="${ZURICH00A_DENSE_GRID_ROWS:-6}"
run_group="${ZURICH00A_DENSE_TARGET_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_sweep_$(date +%Y%m%d_%H%M%S)}"
run_root="${run_group}/veckm_dense_window"
sample_dir="${run_root}/${sequence_label}_density${max_events_per_interval}_p${density_power}_win${window_start}_${window_intervals}"
veckm_out_root="${run_root}/veckm"
veckm_pred_dir="${veckm_out_root}/$(basename "${sample_dir}")"
dense_text_base="${run_group}/dense_window_text"
subset_base="${run_group}/target_car10_dense_base"
build_dir="${ZURICH00A_DENSE_TARGET_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_target_car10_dense_build}"

mkdir -p "${run_group}/logs" "${run_root}"
status_log="${run_group}/logs/status.log"
score_csv="${run_group}/target_scores.csv"
printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_latest_path.txt
printf "run_dir,smooth,label,downsample,smoothness_mode,sigma,min_weight,cand_angle,cand_ratio,initial_candidates,tracked_bonus,track_min,track_max,car_recall,noncar_fpr,car_precision,raw_best_recall,raw_best_purity,raw_label_mean,raw_label_max,nonzero_imo_frames,frames_over_5pct\n" > "${score_csv}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Building dense Zurich 00a target set."
log "Sequence: ${sequence_dir}; window=${window_start}+${window_intervals}; target_frames=${target_frames}"
log "Sampling: density cap=${max_events_per_interval}, power=${density_power}, grid=${grid_cols}x${grid_rows}"

log "Checking Python dependencies."
"${python_bin}" - <<'PY' >"${run_group}/logs/dependency_check.log" 2>&1
import sys
sys.path.insert(0, "/home/DanielH/Optical_Flow/depthanyevent")
import cv2
import h5py
import hdf5plugin
import numpy
import torch
import yaml
from dataset.dsec_dataset.sbt.eventslicer import EventSlicer
from VecKM_flow.inference import VecKMNormalFlowEstimator
print("dependency check ok; cuda:", torch.cuda.is_available())
PY

log "Building EvMotionSeg standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1

log "Sampling dense DSEC window into VecKM arrays."
"${python_bin}" EvMotionSeg/tools/prepare_dsec_veckm_input.py \
  --sequence-dir "${sequence_dir}" \
  --output-dir "${sample_dir}" \
  --start-interval "${window_start}" \
  --num-intervals "${window_intervals}" \
  --interval-s 0.1 \
  --sampling-mode density \
  --max-events-per-interval "${max_events_per_interval}" \
  --grid-cols "${grid_cols}" \
  --grid-rows "${grid_rows}" \
  --density-power "${density_power}" \
  --density-offset "${density_offset}" \
  --preview-stride 1 \
  --overwrite \
  >"${run_group}/logs/prepare_dsec_veckm_input.log" 2>&1

read -r dense_width dense_height dense_fx dense_fy dense_intervals dense_events < <("${python_bin}" - "${sample_dir}/dsec_veckm_input_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)
print(
    summary["width"],
    summary["height"],
    summary["fx"],
    summary["fy"],
    summary["num_intervals"],
    summary["written_events"],
)
PY
)
log "Dense sample ready: intervals=${dense_intervals}, events=${dense_events}, camera=${dense_width}x${dense_height}, fx=${dense_fx}, fy=${dense_fy}"

log "Running VecKM normal-flow inference on dense window."
"${python_bin}" VecKM_flow/run_veckm_evimo2_sliding.py \
  --data-root "$(dirname "${sample_dir}")" \
  --sequences "$(basename "${sample_dir}")" \
  --output-dir "${veckm_out_root}" \
  --training-set DSEC \
  --ensemble 3 \
  --save-undistorted \
  >"${run_group}/logs/veckm.log" 2>&1

log "Rendering VecKM flow masks for the dense window."
"${python_bin}" EvMotionSeg/tools/render_veckm_flow_masks.py \
  "${sample_dir}" \
  "${veckm_pred_dir}" \
  "${run_group}/veckm_flow" \
  --flow-file dataset_pred_flow_DSEC.npy \
  --uncertainty-file dataset_angle_vars_flow_DSEC.npy \
  --timestamp-scale 1e-6 \
  --interval-s 0.1 \
  --num-intervals "${dense_intervals}" \
  --uncertainty-threshold 0.3 \
  >"${run_group}/logs/veckm_flow_masks.log" 2>&1

log "Converting dense VecKM predictions to EvMotionSeg text."
"${python_bin}" EvMotionSeg/tools/prepare_drone_sequence.py text \
  "${sample_dir}" \
  "${veckm_pred_dir}" \
  "${dense_text_base}" \
  --flow-file dataset_pred_flow_DSEC.npy \
  --timestamp-scale 1e-6 \
  --chunk-size 250000 \
  >"${run_group}/logs/evmotionseg_text.log" 2>&1
rm -rf "${dense_text_base}/event_preview"
cp -a "${sample_dir}/event_preview" "${dense_text_base}/event_preview"

log "Extracting target frames into compact 10-frame EvMotionSeg base."
"${python_bin}" EvMotionSeg/tools/prepare_evmotionseg_frame_subset.py \
  --base-run "${dense_text_base}" \
  --output-run "${subset_base}" \
  --frames "${target_frames}" \
  --interval 0.1 \
  >"${run_group}/logs/prepare_target_subset.log" 2>&1

read -r ev_width ev_height ev_fx ev_fy num_intervals < <("${python_bin}" - "${subset_base}/evmotionseg_input_summary.json" "${subset_base}/frame_map.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
frame_map = json.load(open(sys.argv[2], "r", encoding="utf-8"))
print(summary["width"], summary["height"], summary["fx"], summary["fy"], len(frame_map["frames"]))
PY
)
log "Target base ready: width=${ev_width}, height=${ev_height}, fx=${ev_fx}, fy=${ev_fy}, intervals=${num_intervals}."

run_one() {
  local smooth="$1"
  local label="$2"
  local downsample="$3"
  local smooth_mode="$4"
  local sigma="$5"
  local min_weight="$6"
  local cand_angle="$7"
  local cand_ratio="$8"
  local initial_candidates="$9"
  local tracked_bonus="${10}"
  local track_min="${11}"
  local track_max="${12}"
  local name="s${smooth}_l${label}_d${downsample}_${smooth_mode}_sig${sigma}_mw${min_weight}_ang${cand_angle}_r${cand_ratio}_ic${initial_candidates}_tb${tracked_bonus}_tm${track_min}"
  local out="${run_group}/${name}"

  log "Starting ${name}"
  mkdir -p "${out}/logs"
  cp "${subset_base}/events.txt" "${out}/events.txt"
  cp "${subset_base}/flow_xy.txt" "${out}/flow_xy.txt"
  cp "${subset_base}/undistorted_normalized_xy.txt" "${out}/undistorted_normalized_xy.txt"
  cp "${subset_base}/evmotionseg_input_summary.json" "${out}/evmotionseg_input_summary.json"
  cp "${subset_base}/frame_map.json" "${out}/frame_map.json"
  cp "${subset_base}/timestamp.csv" "${out}/timestamp.csv.source_subset"
  cp -a "${subset_base}/event_preview" "${out}/event_preview"

  "${build_dir}/motion_segmentation_standalone" \
    --data_file_path "${out}" \
    --interval 0.1 \
    --width "${ev_width}" \
    --height "${ev_height}" \
    --downsample_rate "${downsample}" \
    --fx "${ev_fx}" \
    --fy "${ev_fy}" \
    --data_term 1 \
    --smooth_term "${smooth}" \
    --label_term "${label}" \
    --GraphCutIteration 10 \
    --MotionSegIteration 4 \
    --max_labels 64 \
    --num_intervals "${num_intervals}" \
    --imo_background_mode background_fit \
    --smoothness_mode "${smooth_mode}" \
    --smoothness_flow_sigma "${sigma}" \
    --smoothness_min_weight "${min_weight}" \
    --candidate_angle_eps "${cand_angle}" \
    --candidate_length_ratio_eps "${cand_ratio}" \
    --initial_candidate_count "${initial_candidates}" \
    --tracked_candidate_bonus "${tracked_bonus}" \
    --label_track_min_fraction "${track_min}" \
    --label_track_max_fraction "${track_max}" \
    >"${out}/logs/evmotionseg.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/summarize_evmotionseg_run.py "${out}" \
    --label "${name}" \
    --stride 1 \
    --max-contact-frames 10 \
    >"${out}/logs/qualitative.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/evaluate_dsec_car_event_coverage.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    >"${out}/logs/car_eval.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/evaluate_dsec_car_raw_label_overlap.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    >"${out}/logs/car_raw_eval.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/render_dsec_car_comparison.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    --frames 0,1,2,3,4,5,6,7,8,9 \
    >"${out}/logs/car_comparison.log" 2>&1

  "${python_bin}" - "${out}" "${score_csv}" "${smooth}" "${label}" "${downsample}" "${smooth_mode}" "${sigma}" "${min_weight}" "${cand_angle}" "${cand_ratio}" "${initial_candidates}" "${tracked_bonus}" "${track_min}" "${track_max}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
score_csv = Path(sys.argv[2])
smooth, label, downsample, smooth_mode, sigma, min_weight, cand_angle, cand_ratio, initial_candidates, tracked_bonus, track_min, track_max = sys.argv[3:15]
name = out.name
qual = json.load(open(out / "qualitative" / f"qualitative_summary_veckm_{name}.json", "r", encoding="utf-8"))
eval_summary = json.load(open(out / "evaluation" / "car_event_coverage_summary.json", "r", encoding="utf-8"))
raw_summary = json.load(open(out / "evaluation" / "car_raw_label_overlap_summary.json", "r", encoding="utf-8"))
with score_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        str(out),
        smooth,
        label,
        downsample,
        smooth_mode,
        sigma,
        min_weight,
        cand_angle,
        cand_ratio,
        initial_candidates,
        tracked_bonus,
        track_min,
        track_max,
        eval_summary.get("micro_car_event_recall"),
        eval_summary.get("micro_noncar_event_fpr"),
        eval_summary.get("micro_car_precision_among_event_moving"),
        raw_summary.get("micro_best_label_car_recall"),
        raw_summary.get("micro_best_label_car_purity"),
        qual.get("raw_label_count_mean"),
        qual.get("raw_label_count_max"),
        qual.get("nonzero_imo_frames"),
        qual.get("frames_over_5pct"),
    ])
PY
  log "Finished ${name}"
}

# Dense 10-frame tuning. The first row is a dense control; the rest bracket the
# cleaner/high-precision and higher-recall settings from the 10k pass.
run_one 7000 90000 1 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 2200 45000 1 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 1800 40000 1 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 3000 45000 1 flow_edge 16 0.10 30 1.5 24 12 0.005 0.35
run_one 2500 40000 1 flow_edge 16 0.10 30 1.5 24 12 0.005 0.35
run_one 2000 35000 1 flow_edge 16 0.10 30 1.5 24 12 0.005 0.35
run_one 1600 35000 1 flow_edge 12 0.10 25 1.4 32 16 0.005 0.35
run_one 1200 30000 1 flow_edge 12 0.10 25 1.4 32 16 0.005 0.35

log "Dense targeted sweep complete. Scores: ${score_csv}"
