#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
subset_base="${ZURICH00A_TARGET_BASE:-EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_base}"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
run_group="${ZURICH00A_TARGET_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_sweep_$(date +%Y%m%d_%H%M%S)}"
build_dir="${ZURICH00A_TARGET_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_target_car10_build}"

mkdir -p "${run_group}/logs"
status_log="${run_group}/logs/status.log"
score_csv="${run_group}/target_scores.csv"
printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_latest_path.txt
printf "run_dir,smooth,label,downsample,smoothness_mode,sigma,min_weight,cand_angle,cand_ratio,initial_candidates,tracked_bonus,track_min,track_max,car_recall,noncar_fpr,car_precision,raw_best_recall,raw_best_purity,raw_label_mean,raw_label_max,nonzero_imo_frames,frames_over_5pct\n" > "${score_csv}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Building standalone binary for targeted Zurich 00a car10 tuning."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1

read -r ev_width ev_height ev_fx ev_fy num_intervals < <("${python_bin}" - "${subset_base}/evmotionseg_input_summary.json" "${subset_base}/frame_map.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
frame_map = json.load(open(sys.argv[2], "r", encoding="utf-8"))
print(summary["width"], summary["height"], summary["fx"], summary["fy"], len(frame_map["frames"]))
PY
)
log "Subset inputs: width=${ev_width} height=${ev_height} fx=${ev_fx} fy=${ev_fy} intervals=${num_intervals}."

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

# Fast targeted sweep on 10 car-heavy failure frames.
run_one 7000 90000 2 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 3500 40000 2 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 1500 40000 2 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 1500 20000 2 constant 25 0.05 45 2.0 12 6 0.02 0.20
run_one 3500 40000 2 flow_edge 12 0.05 45 2.0 12 6 0.02 0.20
run_one 1500 40000 2 flow_edge 12 0.05 45 2.0 12 6 0.02 0.20
run_one 3500 40000 2 flow_edge 12 0.05 25 1.4 24 12 0.005 0.35
run_one 1500 40000 2 flow_edge 12 0.05 25 1.4 24 12 0.005 0.35
run_one 1000 30000 2 flow_edge 12 0.05 25 1.4 32 16 0.005 0.35
run_one 700 20000 2 flow_edge 12 0.05 20 1.25 48 16 0.002 0.50
run_one 1000 30000 1 flow_edge 12 0.05 25 1.4 32 16 0.005 0.35
run_one 700 20000 1 flow_edge 12 0.05 20 1.25 48 16 0.002 0.50

log "Targeted sweep complete. Scores: ${score_csv}"
