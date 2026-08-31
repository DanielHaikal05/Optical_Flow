#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
base_run="${ZURICH00A_BASE_RUN:-EvMotionSeg/data/dsec_zurich_city_00_a_veckm_density10k_p0p75_origparams_down2_bgfit_20260813_101040}"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
run_group="${ZURICH00A_SWEEP_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_car_tuning_$(date +%Y%m%d_%H%M%S)}"
build_dir="${ZURICH00A_SWEEP_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_car_tuning_build}"

mkdir -p "${run_group}/logs"
status_log="${run_group}/logs/status.log"
score_csv="${run_group}/sweep_scores.csv"

printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_car_tuning_latest_path.txt
printf "run_dir,smooth,label,downsample,max_labels,car_recall,noncar_fpr,car_precision,raw_label_mean,raw_label_max,nonzero_imo_frames,frames_over_5pct\n" > "${score_csv}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Building standalone binary for Zurich 00a car tuning."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1

read -r ev_width ev_height ev_fx ev_fy num_intervals < <("${python_bin}" - "${base_run}/evmotionseg_input_summary.json" "${base_run}/timestamp.csv" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
num = sum(1 for line in open(sys.argv[2], "r", encoding="utf-8") if line.strip())
print(summary["width"], summary["height"], summary["fx"], summary["fy"], num)
PY
)
log "Base inputs: width=${ev_width} height=${ev_height} fx=${ev_fx} fy=${ev_fy} intervals=${num_intervals}."

run_one() {
  local smooth="$1"
  local label="$2"
  local downsample="$3"
  local name="smooth${smooth}_label${label}_down${downsample}"
  local out="${run_group}/${name}"

  log "Starting ${name}"
  mkdir -p "${out}/logs"
  ln -s "$(realpath --relative-to="${out}" "${base_run}/events.txt")" "${out}/events.txt"
  ln -s "$(realpath --relative-to="${out}" "${base_run}/flow_xy.txt")" "${out}/flow_xy.txt"
  ln -s "$(realpath --relative-to="${out}" "${base_run}/undistorted_normalized_xy.txt")" "${out}/undistorted_normalized_xy.txt"
  ln -s "$(realpath --relative-to="${out}" "${base_run}/event_preview")" "${out}/event_preview"
  cp "${base_run}/evmotionseg_input_summary.json" "${out}/evmotionseg_input_summary.json"

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
    --max_labels 32 \
    --num_intervals "${num_intervals}" \
    --imo_background_mode background_fit \
    >"${out}/logs/evmotionseg.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/summarize_evmotionseg_run.py "${out}" \
    --label "${name}" \
    --stride 1 \
    --max-contact-frames 24 \
    >"${out}/logs/qualitative.log" 2>&1

  "${python_bin}" EvMotionSeg/tools/evaluate_dsec_car_event_coverage.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    >"${out}/logs/car_eval.log" 2>&1

  "${python_bin}" - "${out}" "${score_csv}" "${smooth}" "${label}" "${downsample}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
score_csv = Path(sys.argv[2])
smooth, label, downsample = sys.argv[3:6]
qual_path = out / "qualitative" / f"qualitative_summary_veckm_smooth{smooth}_label{label}_down{downsample}.json"
eval_path = out / "evaluation" / "car_event_coverage_summary.json"
qual = json.load(open(qual_path, "r", encoding="utf-8"))
eval_summary = json.load(open(eval_path, "r", encoding="utf-8"))
with score_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            str(out),
            smooth,
            label,
            downsample,
            32,
            eval_summary.get("micro_car_event_recall"),
            eval_summary.get("micro_noncar_event_fpr"),
            eval_summary.get("micro_car_precision_among_event_moving"),
            qual.get("raw_label_count_mean"),
            qual.get("raw_label_count_max"),
            qual.get("nonzero_imo_frames"),
            qual.get("frames_over_5pct"),
        ]
    )
PY
  log "Finished ${name}"
}

# Coarse sweep around lower label/smooth costs to encourage car-flow splits.
run_one 7000 60000 2
run_one 7000 40000 2
run_one 7000 20000 2
run_one 3500 60000 2
run_one 3500 40000 2
run_one 3500 20000 2
run_one 1500 40000 2
run_one 1500 20000 2
run_one 7000 20000 1
run_one 3500 20000 1

log "Sweep complete. Scores: ${score_csv}"
