#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

python_bin="${PYTHON_BIN:-/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python}"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
source_base="${ZURICH00A_FULL_BASE:-EvMotionSeg/data/dsec_zurich_city_00_a_veckm_density10k_p0p75_origparams_down2_bgfit_20260813_101040}"
seed_file="${ZURICH00A_SEED_FILE:-EvMotionSeg/data/dsec_zurich_city_00_a_target_car10_dense_sweep_20260813_111455/target_car10_dense_base/evaluation/gt_motion_seed.csv}"
run_group="${ZURICH00A_SEEDED_FULL_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_seeded_full_$(date +%Y%m%d_%H%M%S)}"
out="${run_group}/seeded_data3_s7000_l90000_d1"
build_dir="${ZURICH00A_SEEDED_FULL_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_seeded_full_build}"

mkdir -p "${run_group}/logs" "${out}/logs"
printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_seeded_full_latest_path.txt
status_log="${run_group}/logs/status.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    log "Missing required path: ${path}"
    exit 1
  fi
}

require_file "${source_base}/events.txt"
require_file "${source_base}/flow_xy.txt"
require_file "${source_base}/undistorted_normalized_xy.txt"
require_file "${source_base}/evmotionseg_input_summary.json"
require_file "${source_base}/timestamp.csv"
require_file "${seed_file}"

read -r ev_width ev_height ev_fx ev_fy num_intervals num_events < <("${python_bin}" - "${source_base}/evmotionseg_input_summary.json" "${source_base}/timestamp.csv" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
timestamp_path = Path(sys.argv[2])
num_intervals = sum(1 for line in timestamp_path.read_text(encoding="utf-8").splitlines() if line.strip())
print(
    summary["width"],
    summary["height"],
    summary["fx"],
    summary["fy"],
    num_intervals,
    summary.get("events", ""),
)
PY
)

log "Run group: ${run_group}"
log "Output: ${out}"
log "Source full input: ${source_base}"
log "Seed labels: ${seed_file}"
log "Input: ${num_events} events, ${num_intervals} intervals, ${ev_width}x${ev_height}, fx=${ev_fx}, fy=${ev_fy}"
log "Building standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1

log "Linking read-only inputs and copying metadata."
ln -sfn "$(readlink -f "${source_base}/events.txt")" "${out}/events.txt"
ln -sfn "$(readlink -f "${source_base}/flow_xy.txt")" "${out}/flow_xy.txt"
ln -sfn "$(readlink -f "${source_base}/undistorted_normalized_xy.txt")" "${out}/undistorted_normalized_xy.txt"
if [[ -d "${source_base}/event_preview" ]]; then
  ln -sfn "$(readlink -f "${source_base}/event_preview")" "${out}/event_preview"
fi
cp "${source_base}/evmotionseg_input_summary.json" "${out}/evmotionseg_input_summary.json"
cp "${source_base}/timestamp.csv" "${out}/timestamp.csv.source"
cp "${seed_file}" "${out}/gt_motion_seed.csv"
if [[ -f "${source_base}/frame_map.json" ]]; then
  cp "${source_base}/frame_map.json" "${out}/frame_map.json"
else
  "${python_bin}" - "${num_intervals}" "${out}/frame_map.json" <<'PY'
import json
import sys
from pathlib import Path

num_intervals = int(sys.argv[1])
out = Path(sys.argv[2])
frames = [{"subset_frame": i, "source_frame": i} for i in range(num_intervals)]
out.write_text(json.dumps({"frames": frames}, indent=2) + "\n", encoding="utf-8")
PY
fi

cat >"${out}/command.txt" <<EOF
${build_dir}/motion_segmentation_standalone \\
  --data_file_path "${out}" \\
  --interval 0.1 \\
  --width "${ev_width}" \\
  --height "${ev_height}" \\
  --downsample_rate 1 \\
  --fx "${ev_fx}" \\
  --fy "${ev_fy}" \\
  --data_term 3 \\
  --smooth_term 7000 \\
  --label_term 90000 \\
  --GraphCutIteration 10 \\
  --MotionSegIteration 4 \\
  --max_labels 16 \\
  --num_intervals "${num_intervals}" \\
  --imo_background_mode background_fit \\
  --background_label_error_ratio 3.0 \\
  --background_label_min_fraction 0.10 \\
  --smoothness_mode constant \\
  --smoothness_flow_sigma 25 \\
  --smoothness_min_weight 0.05 \\
  --candidate_angle_eps 45 \\
  --candidate_length_ratio_eps 2.0 \\
  --initial_candidate_count 2 \\
  --tracked_candidate_bonus 0 \\
  --label_track_min_fraction 0.02 \\
  --label_track_max_fraction 0.20 \\
  --label_retention_mode legacy \\
  --seed_labels_file "${seed_file}"
EOF

cat >"${out}/run_config.json" <<EOF
{
  "source_base": "${source_base}",
  "seed_file": "${seed_file}",
  "events": "${num_events}",
  "num_intervals": ${num_intervals},
  "width": ${ev_width},
  "height": ${ev_height},
  "fx": ${ev_fx},
  "fy": ${ev_fy},
  "configuration": {
    "data_term": 3,
    "smooth_term": 7000,
    "label_term": 90000,
    "downsample_rate": 1,
    "graph_cut_iterations": 10,
    "motion_seg_iterations": 4,
    "max_labels": 16,
    "imo_background_mode": "background_fit",
    "background_label_error_ratio": 3.0,
    "background_label_min_fraction": 0.10,
    "smoothness_mode": "constant",
    "smoothness_flow_sigma": 25,
    "smoothness_min_weight": 0.05,
    "candidate_angle_eps": 45,
    "candidate_length_ratio_eps": 2.0,
    "initial_candidate_count": 2,
    "tracked_candidate_bonus": 0,
    "label_track_min_fraction": 0.02,
    "label_track_max_fraction": 0.20,
    "label_retention_mode": "legacy"
  }
}
EOF

log "Starting EvMotionSeg full-sequence seeded run."
"${build_dir}/motion_segmentation_standalone" \
  --data_file_path "${out}" \
  --interval 0.1 \
  --width "${ev_width}" \
  --height "${ev_height}" \
  --downsample_rate 1 \
  --fx "${ev_fx}" \
  --fy "${ev_fy}" \
  --data_term 3 \
  --smooth_term 7000 \
  --label_term 90000 \
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
  --initial_candidate_count 2 \
  --tracked_candidate_bonus 0 \
  --label_track_min_fraction 0.02 \
  --label_track_max_fraction 0.20 \
  --label_retention_mode legacy \
  --seed_labels_file "${seed_file}" \
  >"${out}/logs/evmotionseg.log" 2>&1

log "EvMotionSeg completed. Generating qualitative summaries."
"${python_bin}" EvMotionSeg/tools/summarize_evmotionseg_run.py "${out}" \
  --label "seeded_data3_s7000_l90000_d1" \
  --stride 4 \
  --max-contact-frames 32 \
  >"${out}/logs/qualitative.log" 2>&1

if [[ -d "${sequence_dir}/segmentation/11classes" ]]; then
  log "Running DSEC car-event coverage evaluation."
  "${python_bin}" EvMotionSeg/tools/evaluate_dsec_car_event_coverage.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    >"${out}/logs/car_eval.log" 2>&1

  log "Running raw-label overlap evaluation."
  "${python_bin}" EvMotionSeg/tools/evaluate_dsec_car_raw_label_overlap.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    >"${out}/logs/car_raw_eval.log" 2>&1

  log "Rendering DSEC event/GT/raw/IMO comparison sheet."
  "${python_bin}" EvMotionSeg/tools/render_dsec_car_comparison.py \
    --sequence-dir "${sequence_dir}" \
    --run-dir "${out}" \
    --classes 11 \
    --car-label 8 \
    >"${out}/logs/car_comparison.log" 2>&1
else
  log "Skipping DSEC semantic evaluation; missing ${sequence_dir}/segmentation/11classes."
fi

log "Full-sequence seeded run complete: ${out}"
