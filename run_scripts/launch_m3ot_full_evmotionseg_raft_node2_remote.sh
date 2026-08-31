#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

m3ot_root="${M3OT_ROOT:-Datasets/M3OT}"
modalities_csv="${M3OT_MODALITIES:-rgb,ir}"
splits_csv="${M3OT_SPLITS:-train,val,test}"
max_points_per_frame="${RAFT_EVMOTIONSEG_MAX_POINTS_PER_FRAME:-20000}"
random_fraction="${RAFT_EVMOTIONSEG_RANDOM_FRACTION:-0.25}"
downsample_rate="${EVMOTIONSEG_DOWNSAMPLE_RATE:-1}"
raft_model="${RAFT_EVMOTIONSEG_MODEL:-small}"
raft_updates="${RAFT_EVMOTIONSEG_UPDATES:-12}"
num_intervals="${RAFT_EVMOTIONSEG_NUM_INTERVALS:-0}"
start_frame="${RAFT_EVMOTIONSEG_START_FRAME:-1}"
run_prefix="${EVMOTIONSEG_RUN_PREFIX:-m3ot_full_raft_${raft_model}_${max_points_per_frame}_down${downsample_rate}_bgfit}"
batch_root="EvMotionSeg/data/${run_prefix}_$(date +%Y%m%d_%H%M%S)"
latest_link="EvMotionSeg/data/${run_prefix}_latest"
latest_path_file="EvMotionSeg/data/${run_prefix}_latest_path.txt"
status_log="${batch_root}/logs/status.log"
sequence_status="${batch_root}/logs/sequence_status.jsonl"
python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
build_dir="/tmp/m3ot_full_evmotionseg_raft_build"

mkdir -p "${batch_root}/logs"
ln -sfn "$(basename "${batch_root}")" "${latest_link}"
printf '%s\n' "${batch_root}" > "${latest_path_file}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

json_status() {
  "${python_bin}" - "$sequence_status" "$@" <<'PY'
import json
import sys
path = sys.argv[1]
items = sys.argv[2:]
record = {}
for item in items:
    key, value = item.split("=", 1)
    record[key] = value
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

sanitize_label() {
  printf '%s' "$1" | sed 's#^Datasets/M3OT/##; s#[^A-Za-z0-9_.-]#_#g'
}

read_summary_value() {
  "${python_bin}" - "$1" "$2" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data[sys.argv[2]])
PY
}

log "Started full M3OT RAFT -> EvMotionSeg evaluation on $(hostname)."
log "Batch root: ${batch_root}"
log "M3OT root: ${m3ot_root}"
log "Modalities: ${modalities_csv}; splits: ${splits_csv}"
log "Intervals per sequence: ${num_intervals} (0 means all frame pairs); start_frame=${start_frame}"
log "Sampling: max ${max_points_per_frame} pseudo-events/frame; random_fraction=${random_fraction}; downsample_rate=${downsample_rate}"
log "RAFT: model=${raft_model}, updates=${raft_updates}"

log "Checking Python dependencies."
"${python_bin}" - <<'PY' >"${batch_root}/logs/dependency_check.log" 2>&1
import cv2
import numpy
import torch
import torchvision
from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available())
print("dependency check ok")
PY

log "Building EvMotionSeg portable standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" \
  >"${batch_root}/logs/build_standalone.log" 2>&1

IFS=',' read -r -a modalities <<<"${modalities_csv}"
IFS=',' read -r -a splits <<<"${splits_csv}"
sequences=()
for group in 1 2; do
  for modality in "${modalities[@]}"; do
    for split in "${splits[@]}"; do
      base="${m3ot_root}/${group}/${modality}/${split}"
      if [ -d "${base}" ]; then
        while IFS= read -r seq_dir; do
          sequences+=("${seq_dir}")
        done < <(find "${base}" -mindepth 1 -maxdepth 1 -type d | sort)
      fi
    done
  done
done

if [ "${#sequences[@]}" -eq 0 ]; then
  log "No sequences found."
  exit 1
fi

printf '%s\n' "${sequences[@]}" >"${batch_root}/logs/sequences.txt"
log "Found ${#sequences[@]} sequences."

completed=0
failed=0
for seq_dir in "${sequences[@]}"; do
  rel="${seq_dir#${m3ot_root}/}"
  label="$(sanitize_label "${seq_dir}")"
  out_dir="${batch_root}/sequences/${label}"
  mkdir -p "${out_dir}/logs"
  log "Sequence start: ${rel} -> ${out_dir}"
  json_status status=started sequence="${rel}" run_dir="${out_dir}" timestamp="$(date -Iseconds)"

  if [ -f "${out_dir}/evaluation_m3ot.json" ]; then
    log "Sequence already evaluated, skipping: ${rel}"
    completed=$((completed + 1))
    json_status status=skipped sequence="${rel}" run_dir="${out_dir}" timestamp="$(date -Iseconds)"
    continue
  fi

  set +e
  "${python_bin}" EvMotionSeg/tools/prepare_m3ot_raft_for_evmotionseg.py \
    --sequence-dir "${seq_dir}" \
    --output-dir "${out_dir}" \
    --start-frame "${start_frame}" \
    --num-intervals "${num_intervals}" \
    --max-points-per-frame "${max_points_per_frame}" \
    --random-fraction "${random_fraction}" \
    --preview-stride 25 \
    --raft-model "${raft_model}" \
    --raft-updates "${raft_updates}" \
    >"${out_dir}/logs/prepare_raft.log" 2>&1
  prep_rc=$?
  set -e
  if [ "${prep_rc}" -ne 0 ]; then
    log "Sequence failed during RAFT prep: ${rel}"
    failed=$((failed + 1))
    json_status status=failed stage=prepare sequence="${rel}" run_dir="${out_dir}" timestamp="$(date -Iseconds)"
    continue
  fi

  summary="${out_dir}/evmotionseg_input_summary.json"
  prepared_intervals="$(read_summary_value "${summary}" num_intervals)"
  interval_s="$(read_summary_value "${summary}" interval_s)"
  width="$(read_summary_value "${summary}" width)"
  height="$(read_summary_value "${summary}" height)"
  fx="$(read_summary_value "${summary}" fx)"
  fy="$(read_summary_value "${summary}" fy)"

  set +e
  "${build_dir}/motion_segmentation_standalone" \
    --data_file_path "${out_dir}" \
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
    >"${out_dir}/logs/evmotionseg.log" 2>&1
  ev_rc=$?
  set -e
  if [ "${ev_rc}" -ne 0 ]; then
    log "Sequence failed during EvMotionSeg: ${rel}"
    failed=$((failed + 1))
    json_status status=failed stage=evmotionseg sequence="${rel}" run_dir="${out_dir}" timestamp="$(date -Iseconds)"
    continue
  fi

  set +e
  "${python_bin}" EvMotionSeg/tools/evaluate_m3ot_evmotionseg.py \
    --sequence-dir "${seq_dir}" \
    --run-dir "${out_dir}" \
    --output-json "${out_dir}/evaluation_m3ot.json" \
    >"${out_dir}/logs/evaluation.log" 2>&1
  eval_rc=$?
  set -e
  if [ "${eval_rc}" -ne 0 ]; then
    log "Sequence failed during evaluation: ${rel}"
    failed=$((failed + 1))
    json_status status=failed stage=evaluation sequence="${rel}" run_dir="${out_dir}" timestamp="$(date -Iseconds)"
    continue
  fi

  "${python_bin}" EvMotionSeg/tools/summarize_evmotionseg_run.py "${out_dir}" \
    --label "${label}" \
    --stride 25 \
    --max-contact-frames 16 \
    >"${out_dir}/logs/qualitative.log" 2>&1 || true

  completed=$((completed + 1))
  log "Sequence complete: ${rel} (${completed}/${#sequences[@]}, failed=${failed})"
  json_status status=complete sequence="${rel}" run_dir="${out_dir}" timestamp="$(date -Iseconds)"
done

log "Aggregating metrics."
"${python_bin}" - "${batch_root}" <<'PY' >"${batch_root}/logs/aggregate.log" 2>&1
import json
import sys
from pathlib import Path

batch = Path(sys.argv[1])
eval_paths = sorted((batch / "sequences").glob("*/evaluation_m3ot.json"))
items = [json.loads(path.read_text(encoding="utf-8")) for path in eval_paths]

def s(key):
    return sum(float(item.get(key, 0.0) or 0.0) for item in items)

total_pred = s("total_pred_pixels")
total_gt = s("total_gt_pixels")
total_intersection = s("total_intersection_pixels")
total_union = sum(
    (float(item.get("total_pred_pixels", 0) or 0)
     + float(item.get("total_gt_pixels", 0) or 0)
     - float(item.get("total_intersection_pixels", 0) or 0))
    for item in items
)
total_boxes = s("total_boxes")
total_hit_boxes = s("total_hit_boxes")
total_any_hit_boxes = s("total_any_hit_boxes")
frames_with_gt = s("frames_with_gt")
frames_with_gt_hit = s("frames_with_gt_hit")
num_masks = s("num_masks")
moving_weighted = sum(float(item.get("moving_fraction", 0) or 0) * float(item.get("num_masks", 0) or 0) for item in items)

summary = {
    "batch_root": str(batch),
    "num_sequences_evaluated": len(items),
    "num_masks": int(num_masks),
    "total_boxes": int(total_boxes),
    "total_hit_boxes": int(total_hit_boxes),
    "total_any_hit_boxes": int(total_any_hit_boxes),
    "pixel_precision": total_intersection / total_pred if total_pred else 0.0,
    "pixel_recall": total_intersection / total_gt if total_gt else 0.0,
    "pixel_iou": total_intersection / total_union if total_union else 0.0,
    "box_recall_at_threshold": total_hit_boxes / total_boxes if total_boxes else 0.0,
    "box_recall_any_overlap": total_any_hit_boxes / total_boxes if total_boxes else 0.0,
    "frame_hit_rate": frames_with_gt_hit / frames_with_gt if frames_with_gt else 0.0,
    "moving_fraction_mean_weighted_by_frames": moving_weighted / num_masks if num_masks else 0.0,
    "per_sequence": items,
}
(batch / "evaluation_m3ot_full_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

log "Full M3OT run complete. completed=${completed}, failed=${failed}"
