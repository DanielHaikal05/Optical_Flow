#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

python_bin="${PYTHON_BIN:-/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python}"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
dense_base="${ZURICH00A_DENSE_BASE:-EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_sweep_20260813_111455/target_car10_dense_base}"
seed_file="${ZURICH00A_SEED_FILE:-${dense_base}/evaluation/gt_motion_seed.csv}"
run_group="${ZURICH00A_SEEDED_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_seeded_frame4_$(date +%Y%m%d_%H%M%S)}"
subset_base="${run_group}/frame4_base"
build_dir="${ZURICH00A_SEEDED_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_seeded_frame4_build}"

mkdir -p "${run_group}/logs"
printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_seeded_frame4_latest_path.txt
status_log="${run_group}/logs/status.log"
score_csv="${run_group}/target_scores.csv"
printf "run_dir,seeded,data,smooth,label,downsample,car_recall,noncar_fpr,car_precision,raw_best_recall,raw_best_purity,raw_label_mean,raw_label_max\n" > "${score_csv}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Building standalone binary for seeded frame-4 test."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1

log "Extracting representative subset frame 4."
"${python_bin}" EvMotionSeg/tools/prepare_evmotionseg_frame_subset.py \
  --base-run "${dense_base}" \
  --output-run "${subset_base}" \
  --frames 4 \
  --interval 0.1 \
  >"${run_group}/logs/prepare_subset.log" 2>&1

read -r ev_width ev_height ev_fx ev_fy num_intervals < <("${python_bin}" - "${subset_base}/evmotionseg_input_summary.json" "${subset_base}/frame_map.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
frame_map = json.load(open(sys.argv[2], "r", encoding="utf-8"))
print(summary["width"], summary["height"], summary["fx"], summary["fy"], len(frame_map["frames"]))
PY
)
log "Base: ${subset_base}; ${num_intervals} frame, ${ev_width}x${ev_height}."

run_one() {
  local seeded="$1"
  local data="$2"
  local smooth="$3"
  local label="$4"
  local downsample="$5"
  local name="${seeded}_data${data}_s${smooth}_l${label}_d${downsample}"
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

  seed_args=()
  if [[ "${seeded}" == "seeded" ]]; then
    seed_args=(--seed_labels_file "${seed_file}" --initial_candidate_count 2 --tracked_candidate_bonus 0)
  else
    seed_args=(--initial_candidate_count 12 --tracked_candidate_bonus 6)
  fi

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
    --max_labels 16 \
    --num_intervals "${num_intervals}" \
    --imo_background_mode background_fit \
    --background_label_error_ratio 3.0 \
    --background_label_min_fraction 0.10 \
    --smoothness_mode constant \
    --smoothness_flow_sigma 25 \
    --smoothness_min_weight 0.05 \
    --candidate_angle_eps 45 \
    --candidate_length_ratio_eps 2.0 \
    --label_track_min_fraction 0.02 \
    --label_track_max_fraction 0.20 \
    --label_retention_mode legacy \
    "${seed_args[@]}" \
    >"${out}/logs/evmotionseg.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/summarize_evmotionseg_run.py "${out}" \
    --label "${name}" \
    --stride 1 \
    --max-contact-frames 1 \
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
    --frames 0 \
    >"${out}/logs/car_comparison.log" 2>&1

  "${python_bin}" - "${out}" "${score_csv}" "${seeded}" "${data}" "${smooth}" "${label}" "${downsample}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
score_csv = Path(sys.argv[2])
values = sys.argv[3:8]
name = out.name
qual = json.load(open(out / "qualitative" / f"qualitative_summary_veckm_{name}.json", "r", encoding="utf-8"))
eval_summary = json.load(open(out / "evaluation" / "car_event_coverage_summary.json", "r", encoding="utf-8"))
raw_summary = json.load(open(out / "evaluation" / "car_raw_label_overlap_summary.json", "r", encoding="utf-8"))
with score_csv.open("a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(
        [
            str(out),
            *values,
            eval_summary.get("micro_car_event_recall"),
            eval_summary.get("micro_noncar_event_fpr"),
            eval_summary.get("micro_car_precision_among_event_moving"),
            raw_summary.get("micro_best_label_car_recall"),
            raw_summary.get("micro_best_label_car_purity"),
            qual.get("raw_label_count_mean"),
            qual.get("raw_label_count_max"),
        ]
    )
PY
  log "Finished ${name}"
}

run_one unseeded 1 7000 90000 1
run_one seeded 1 7000 90000 1
run_one seeded 3 7000 90000 1
run_one seeded 5 7000 90000 1
run_one seeded 10 7000 90000 1

log "Seeded frame-4 test complete: ${score_csv}"
