#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

export PYTHONPATH="/home/DanielH/Optical_Flow/VecKM_flow:${PYTHONPATH:-}"

python_bin="${PYTHON_BIN:-/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python}"
build_dir="${EVMOTIONSEG_BUILD_DIR:-/tmp/dsec_evmotionseg_veckm_evimo_all_build}"
downsample_rate="${EVMOTIONSEG_DOWNSAMPLE_RATE:-1}"
smooth_term="${EVMOTIONSEG_SMOOTH_TERM:-6000}"
label_term="${EVMOTIONSEG_LABEL_TERM:-60000}"
max_labels="${EVMOTIONSEG_MAX_LABELS:-24}"
preview_stride="${EVMOTIONSEG_PREVIEW_STRIDE:-4}"
imo_background_mode="${EVMOTIONSEG_IMO_BACKGROUND_MODE:-background_fit}"
interval_s="${EVMOTIONSEG_INTERVAL_S:-0.1}"
ensemble="${VECKM_ENSEMBLE:-3}"
training_set="EVIMO"

if [[ -n "${DSEC_SEQUENCE_DIRS:-}" ]]; then
  read -r -a sequences <<<"${DSEC_SEQUENCE_DIRS}"
else
  sequences=(
    "Datasets/DSEC/zurich_city_00_a"
    "Datasets/DSEC/interlaken_00_c"
    "Datasets/DSEC/zurich_city_00_b"
  )
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
group_root="EvMotionSeg/data/dsec_veckm_evimo_all3_${timestamp}"
group_status_log="${group_root}/status.log"
mkdir -p "${group_root}"

log_group() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${group_status_log}"
}

log() {
  local status_log="$1"
  shift
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}" "${group_status_log}"
}

log_group "Started DSEC VecKM-${training_set} -> EvMotionSeg all-events group on $(hostname)."
log_group "Group output: ${group_root}"
log_group "Sequences: ${sequences[*]}"
log_group "EvMotionSeg terms: data=1, smooth=${smooth_term}, label=${label_term}, max_labels=${max_labels}, imo_background_mode=${imo_background_mode}"

log_group "Checking Python dependencies."
"${python_bin}" - <<'PY' >>"${group_root}/dependency_check.log" 2>&1
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
estimator = VecKMNormalFlowEstimator(training_set="EVIMO")
print("VecKM EVIMO model ready")
PY

log_group "Building EvMotionSeg portable standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" \
  >"${group_root}/build_standalone.log" 2>&1

for sequence_dir in "${sequences[@]}"; do
  sequence_label="$(basename "${sequence_dir}")"
  run_name="dsec_${sequence_label}_veckm_evimo_all_down${downsample_rate}_bgfit_${timestamp}"
  run_root="VecKM_flow/outputs/${run_name}"
  sample_dir="${run_root}/${sequence_label}_all"
  veckm_out_root="${run_root}/veckm"
  veckm_pred_dir="${veckm_out_root}/$(basename "${sample_dir}")"
  ev_out="EvMotionSeg/data/${run_name}"
  status_log="${ev_out}/logs/status.log"

  mkdir -p "${ev_out}/logs" "${run_root}"
  ln -sfn "$(basename "${ev_out}")" "EvMotionSeg/data/dsec_${sequence_label}_veckm_evimo_all_latest"
  printf '%s\n' "${ev_out}" >"EvMotionSeg/data/dsec_${sequence_label}_veckm_evimo_all_latest_path.txt"
  ln -sfn "$(basename "${run_root}")" "VecKM_flow/outputs/dsec_${sequence_label}_veckm_evimo_all_latest"
  printf '%s\n' "${run_root}" >"VecKM_flow/outputs/dsec_${sequence_label}_veckm_evimo_all_latest_path.txt"

  log "${status_log}" "Started ${sequence_label}."
  log "${status_log}" "Sequence: ${sequence_dir}"
  log "${status_log}" "Output directory: ${ev_out}"
  log "${status_log}" "VecKM run root: ${run_root}"
  log "${status_log}" "Sampling mode: none; max_events_per_interval=0 keeps all valid rectified events."
  log "${status_log}" "Normal-flow source: VecKM ${training_set} checkpoint."

  log "${status_log}" "Preparing DSEC all-events VecKM input arrays."
  "${python_bin}" EvMotionSeg/tools/prepare_dsec_veckm_input.py \
    --sequence-dir "${sequence_dir}" \
    --output-dir "${sample_dir}" \
    --interval-s "${interval_s}" \
    --sampling-mode global \
    --max-events-per-interval 0 \
    --preview-stride "${preview_stride}" \
    --overwrite \
    >"${ev_out}/logs/prepare_dsec_veckm_input.log" 2>&1

  num_intervals="$("${python_bin}" - "${sample_dir}/dsec_veckm_input_summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["num_intervals"])
PY
)"
  written_events="$("${python_bin}" - "${sample_dir}/dsec_veckm_input_summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["written_events"])
PY
)"
  read -r ev_width ev_height ev_fx ev_fy < <("${python_bin}" - "${sample_dir}/dsec_veckm_input_summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)
print(summary["width"], summary["height"], summary["fx"], summary["fy"])
PY
)
  log "${status_log}" "Prepared ${num_intervals} intervals with ${written_events} valid rectified events."
  log "${status_log}" "Rectified event camera: width=${ev_width}, height=${ev_height}, fx=${ev_fx}, fy=${ev_fy}"

  log "${status_log}" "Running VecKM-${training_set} normal-flow inference."
  "${python_bin}" VecKM_flow/run_veckm_evimo2_sliding.py \
    --data-root "$(dirname "${sample_dir}")" \
    --sequences "$(basename "${sample_dir}")" \
    --output-dir "${veckm_out_root}" \
    --training-set "${training_set}" \
    --ensemble "${ensemble}" \
    --save-undistorted \
    >"${ev_out}/logs/veckm.log" 2>&1

  log "${status_log}" "Rendering VecKM flow masks."
  "${python_bin}" EvMotionSeg/tools/render_veckm_flow_masks.py \
    "${sample_dir}" \
    "${veckm_pred_dir}" \
    "${ev_out}/veckm_flow" \
    --flow-file "dataset_pred_flow_${training_set}.npy" \
    --uncertainty-file "dataset_angle_vars_flow_${training_set}.npy" \
    --timestamp-scale 1e-6 \
    --interval-s "${interval_s}" \
    --num-intervals "${num_intervals}" \
    --uncertainty-threshold 0.3 \
    >"${ev_out}/logs/veckm_flow_masks.log" 2>&1

  log "${status_log}" "Converting VecKM predictions into EvMotionSeg text inputs."
  "${python_bin}" EvMotionSeg/tools/prepare_drone_sequence.py text \
    "${sample_dir}" \
    "${veckm_pred_dir}" \
    "${ev_out}" \
    --flow-file "dataset_pred_flow_${training_set}.npy" \
    --timestamp-scale 1e-6 \
    --chunk-size 250000 \
    >"${ev_out}/logs/evmotionseg_text.log" 2>&1
  rm -rf "${ev_out}/event_preview"
  cp -a "${sample_dir}/event_preview" "${ev_out}/event_preview"

  log "${status_log}" "Starting EvMotionSeg."
  "${build_dir}/motion_segmentation_standalone" \
    --data_file_path "${ev_out}" \
    --interval "${interval_s}" \
    --width "${ev_width}" \
    --height "${ev_height}" \
    --downsample_rate "${downsample_rate}" \
    --fx "${ev_fx}" \
    --fy "${ev_fy}" \
    --data_term 1 \
    --smooth_term "${smooth_term}" \
    --label_term "${label_term}" \
    --GraphCutIteration 10 \
    --MotionSegIteration 4 \
    --max_labels "${max_labels}" \
    --num_intervals "${num_intervals}" \
    --imo_background_mode "${imo_background_mode}" \
    >"${ev_out}/logs/evmotionseg.log" 2>&1

  log "${status_log}" "Generating qualitative summary."
  "${python_bin}" - "${ev_out}" "${training_set}" <<'PY' >"${ev_out}/logs/qualitative.log" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = Path(sys.argv[1])
training_set = sys.argv[2]
preview_dir = base / "event_preview"
raw_results_dir = base / "results"
imo_results_dir = base / "results_imo"
out_dir = base / "qualitative"
out_dir.mkdir(exist_ok=True)

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
small_font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()


def moving_mask(mask_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path).convert("RGB"))
    return np.any(arr < 200, axis=2)


def overlay_imo(idx: int) -> tuple[Image.Image, float] | None:
    preview_path = preview_dir / f"{idx:06d}.png"
    mask_path = imo_results_dir / f"{idx}.png"
    if not preview_path.exists() or not mask_path.exists():
        return None
    image = np.asarray(Image.open(preview_path).convert("RGB")).astype(np.float32)
    mask = moving_mask(mask_path)
    image[mask] = image[mask] * 0.25 + np.array([0, 255, 70], dtype=np.float32) * 0.75
    out = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    frac = float(mask.mean())
    label = f"t={idx * 0.1:05.1f}s  frame {idx:03d}  IMO {frac * 100:.2f}%"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, bbox[2] + 20, bbox[3] + 18), fill=(0, 0, 0))
    draw.text((14, 11), label, fill=(255, 255, 255), font=font)
    return out, frac


def overlay_raw(idx: int) -> tuple[Image.Image, int] | None:
    preview_path = preview_dir / f"{idx:06d}.png"
    mask_path = raw_results_dir / f"{idx}.png"
    if not preview_path.exists() or not mask_path.exists():
        return None
    preview = np.asarray(Image.open(preview_path).convert("RGB")).astype(np.float32)
    labels = np.asarray(Image.open(mask_path).convert("RGB"))
    foreground = np.any(labels < 245, axis=2)
    preview[foreground] = preview[foreground] * 0.35 + labels[foreground].astype(np.float32) * 0.65
    out = Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8))
    colors = np.unique(labels[foreground].reshape(-1, 3), axis=0) if np.any(foreground) else np.empty((0, 3))
    label_count = int(colors.shape[0])
    draw = ImageDraw.Draw(out)
    label = f"t={idx * 0.1:05.1f}s  frame {idx:03d}  labels {label_count}"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, bbox[2] + 20, bbox[3] + 18), fill=(0, 0, 0))
    draw.text((14, 11), label, fill=(255, 255, 255), font=font)
    return out, label_count


def write_contact_sheet(overlays: list[tuple[int, Image.Image]], path: Path) -> None:
    sample_count = min(24, len(overlays))
    if sample_count == 0:
        return
    sample_positions = np.linspace(0, len(overlays) - 1, sample_count).round().astype(int)
    samples = [overlays[i] for i in sample_positions]
    tile_w, tile_h = samples[0][1].size
    cols = 4
    rows = int(np.ceil(sample_count / cols))
    margin = 18
    title_h = 38
    sheet = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * margin, rows * (tile_h + title_h) + (rows + 1) * margin),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for n, (idx, image) in enumerate(samples):
        row, col = divmod(n, cols)
        x = margin + col * (tile_w + margin)
        y = margin + row * (tile_h + title_h + margin)
        draw.text((x, y), f"{idx:03d} / {idx * 0.1:.1f}s", fill=(230, 230, 230), font=small_font)
        sheet.paste(image, (x, y + title_h))
    sheet.save(path)


def write_video(overlays: list[tuple[int, Image.Image]], path: Path) -> None:
    if not overlays:
        return
    tile_w, tile_h = overlays[0][1].size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (tile_w, tile_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for _, image in overlays:
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()


all_imo_mask_files = sorted(imo_results_dir.glob("*.png"), key=lambda path: int(path.stem))
all_fracs = [float(moving_mask(path).mean()) for path in all_imo_mask_files]
imo_overlays = []
raw_overlays = []
raw_label_counts = []
for preview_path in sorted(preview_dir.glob("*.png"), key=lambda path: int(path.stem)):
    idx = int(preview_path.stem)
    item = overlay_imo(idx)
    if item is not None:
        imo_overlays.append((idx, item[0]))
    raw_item = overlay_raw(idx)
    if raw_item is not None:
        raw_overlays.append((idx, raw_item[0]))
        raw_label_counts.append((idx, raw_item[1]))

contact_sheet = out_dir / f"dsec_evmotionseg_veckm_{training_set.lower()}_all_imo_contact_sheet.png"
video_path = out_dir / f"dsec_evmotionseg_veckm_{training_set.lower()}_all_imo_stride4.mp4"
raw_contact_sheet = out_dir / f"dsec_evmotionseg_veckm_{training_set.lower()}_all_raw_contact_sheet.png"
raw_video_path = out_dir / f"dsec_evmotionseg_veckm_{training_set.lower()}_all_raw_stride4.mp4"
write_contact_sheet(imo_overlays, contact_sheet)
write_video(imo_overlays, video_path)
write_contact_sheet(raw_overlays, raw_contact_sheet)
write_video(raw_overlays, raw_video_path)

arr = np.array(all_fracs, dtype=np.float64)
raw_counts_arr = np.array([count for _, count in raw_label_counts], dtype=np.float64)
summary = {
    "run_dir": str(base),
    "normal_flow_training_set": training_set,
    "sampling": "none",
    "num_masks": int(len(all_fracs)),
    "num_preview_overlays": int(len(imo_overlays)),
    "num_raw_preview_overlays": int(len(raw_overlays)),
    "moving_fraction_mean": float(np.mean(arr)) if arr.size else None,
    "moving_fraction_median": float(np.median(arr)) if arr.size else None,
    "moving_fraction_p90": float(np.percentile(arr, 90)) if arr.size else None,
    "moving_fraction_max": float(np.max(arr)) if arr.size else None,
    "nonzero_imo_frames": int(np.count_nonzero(arr > 0)) if arr.size else 0,
    "frames_over_1pct": int(np.count_nonzero(arr > 0.01)) if arr.size else 0,
    "frames_over_5pct": int(np.count_nonzero(arr > 0.05)) if arr.size else 0,
    "top_frames": [[int(all_imo_mask_files[i].stem), float(arr[i])] for i in np.argsort(arr)[-10:][::-1]] if arr.size else [],
    "contact_sheet": str(contact_sheet),
    "video": str(video_path),
    "raw_label_count_mean": float(np.mean(raw_counts_arr)) if raw_counts_arr.size else None,
    "raw_label_count_median": float(np.median(raw_counts_arr)) if raw_counts_arr.size else None,
    "raw_label_count_max": int(np.max(raw_counts_arr)) if raw_counts_arr.size else None,
    "raw_top_label_count_frames": sorted(raw_label_counts, key=lambda item: item[1], reverse=True)[:10],
    "raw_contact_sheet": str(raw_contact_sheet),
    "raw_video": str(raw_video_path),
}
(out_dir / "qualitative_summary_veckm_evimo_all.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

  "${python_bin}" - "${sample_dir}/dsec_veckm_input_summary.json" "${veckm_pred_dir}/summary.json" "${ev_out}/qualitative/qualitative_summary_veckm_evimo_all.json" "${ev_out}/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path

sample_summary = json.loads(Path(sys.argv[1]).read_text())
veckm_summary = json.loads(Path(sys.argv[2]).read_text())
qual_summary = json.loads(Path(sys.argv[3]).read_text())
summary = {
    "sequence": Path(sample_summary["sequence_dir"]).name,
    "sequence_dir": sample_summary["sequence_dir"],
    "run_dir": str(Path(sys.argv[4]).parent),
    "normal_flow_training_set": "EVIMO",
    "sampling": "none",
    "interval_s": sample_summary["interval_s"],
    "num_intervals": sample_summary["num_intervals"],
    "written_events": sample_summary["written_events"],
    "mean_written_events_per_interval": sample_summary["mean_written_events_per_interval"],
    "veckm": veckm_summary,
    "qualitative": qual_summary,
}
Path(sys.argv[4]).write_text(json.dumps(summary, indent=2) + "\n")
PY

  cp "${ev_out}/run_summary.json" "${group_root}/${sequence_label}_run_summary.json"
  log "${status_log}" "Completed ${sequence_label}."
done

"${python_bin}" - "${group_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = []
for path in sorted(root.glob("*_run_summary.json")):
    items.append(json.loads(path.read_text()))
summary = {
    "group_root": str(root),
    "normal_flow_training_set": "EVIMO",
    "sampling": "none",
    "sequences": items,
    "total_written_events": sum(item["written_events"] for item in items),
    "total_intervals": sum(item["num_intervals"] for item in items),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

log_group "DSEC VecKM-${training_set} -> EvMotionSeg all-events group complete."
