#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
latest_dense_group="$(cat EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_latest_path.txt)"
subset_base="${ZURICH00A_DENSE_TARGET_BASE:-${latest_dense_group}/target_car10_dense_base}"
run_group="${ZURICH00A_DENSE_DATA_REFINE_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_data_refine_$(date +%Y%m%d_%H%M%S)}"
build_dir="${ZURICH00A_DENSE_DATA_REFINE_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_dense_data_refine_build}"

mkdir -p "${run_group}/logs"
status_log="${run_group}/logs/status.log"
score_csv="${run_group}/target_scores.csv"
printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_data_refine_latest_path.txt
printf "run_dir,data,smooth,label,downsample,background_mode,bg_error_ratio,bg_min_fraction,smoothness_mode,sigma,min_weight,cand_angle,cand_ratio,initial_candidates,tracked_bonus,track_min,track_max,car_recall,noncar_fpr,car_precision,raw_best_recall,raw_best_purity,raw_label_mean,raw_label_max,nonzero_imo_frames,frames_over_5pct\n" > "${score_csv}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Building standalone binary for dense-base data-term refinement."
log "Subset base: ${subset_base}"
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1

read -r ev_width ev_height ev_fx ev_fy num_intervals < <("${python_bin}" - "${subset_base}/evmotionseg_input_summary.json" "${subset_base}/frame_map.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
frame_map = json.load(open(sys.argv[2], "r", encoding="utf-8"))
print(summary["width"], summary["height"], summary["fx"], summary["fy"], len(frame_map["frames"]))
PY
)
log "Dense target base: width=${ev_width} height=${ev_height} fx=${ev_fx} fy=${ev_fy} intervals=${num_intervals}."

run_one() {
  local data="$1"
  local smooth="$2"
  local label="$3"
  local downsample="$4"
  local background_mode="$5"
  local bg_error_ratio="$6"
  local bg_min_fraction="$7"
  local smooth_mode="$8"
  local sigma="$9"
  local min_weight="${10}"
  local cand_angle="${11}"
  local cand_ratio="${12}"
  local initial_candidates="${13}"
  local tracked_bonus="${14}"
  local track_min="${15}"
  local track_max="${16}"
  local name="data${data}_s${smooth}_l${label}_d${downsample}_${background_mode}_bgr${bg_error_ratio}_${smooth_mode}_sig${sigma}_mw${min_weight}_ang${cand_angle}_r${cand_ratio}"
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
    --data_term "${data}" \
    --smooth_term "${smooth}" \
    --label_term "${label}" \
    --GraphCutIteration 10 \
    --MotionSegIteration 4 \
    --max_labels 64 \
    --num_intervals "${num_intervals}" \
    --imo_background_mode "${background_mode}" \
    --background_label_error_ratio "${bg_error_ratio}" \
    --background_label_min_fraction "${bg_min_fraction}" \
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

  "${python_bin}" - "${out}" "${score_csv}" "${data}" "${smooth}" "${label}" "${downsample}" "${background_mode}" "${bg_error_ratio}" "${bg_min_fraction}" "${smooth_mode}" "${sigma}" "${min_weight}" "${cand_angle}" "${cand_ratio}" "${initial_candidates}" "${tracked_bonus}" "${track_min}" "${track_max}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
score_csv = Path(sys.argv[2])
values = sys.argv[3:19]
name = out.name
qual = json.load(open(out / "qualitative" / f"qualitative_summary_veckm_{name}.json", "r", encoding="utf-8"))
eval_summary = json.load(open(out / "evaluation" / "car_event_coverage_summary.json", "r", encoding="utf-8"))
raw_summary = json.load(open(out / "evaluation" / "car_raw_label_overlap_summary.json", "r", encoding="utf-8"))
with score_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        str(out),
        *values,
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

run_one 3 12000 180000 1 background_fit_multi 1.3 0.04 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 5 16000 240000 1 background_fit_multi 1.3 0.04 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 5 14000 220000 1 background_fit_multi 1.2 0.04 flow_edge 20 0.20 35 1.7 18 8 0.01 0.25

log "Dense-base data-term refinement complete. Scores: ${score_csv}"
