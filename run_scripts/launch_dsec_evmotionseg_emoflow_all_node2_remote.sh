#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

max_events_per_interval="${EVMOTIONSEG_MAX_EVENTS_PER_INTERVAL:-0}"
downsample_rate="${EVMOTIONSEG_DOWNSAMPLE_RATE:-1}"
run_prefix="${EVMOTIONSEG_RUN_PREFIX:-dsec_interlaken_00_c_emoflow_all_events_down${downsample_rate}_bgfit}"
run_name="${run_prefix}_$(date +%Y%m%d_%H%M%S)"
ev_out="EvMotionSeg/data/${run_name}"
latest_link="EvMotionSeg/data/${run_prefix}_latest"
latest_path_file="EvMotionSeg/data/${run_prefix}_latest_path.txt"
status_log="${ev_out}/logs/status.log"
python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
build_dir="/tmp/dsec_evmotionseg_full_build"
sequence_dir="Datasets/DSEC/interlaken_00_c"
flow_dir="E-MoFlow/outputs/dsec_local_egomotion/interlaken_00_c_dgx_node2_full_20260807_1806/submission_pred_flow"

mkdir -p "${ev_out}/logs"
ln -sfn "$(basename "${ev_out}")" "${latest_link}"
printf '%s\n' "${ev_out}" > "${latest_path_file}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Started full-event DSEC EvMotionSeg run on $(hostname)."
log "Output directory: ${ev_out}"
log "Sequence: ${sequence_dir}"
log "EMoFlow predicted flow: ${flow_dir}"
if [ "${max_events_per_interval}" = "0" ]; then
  event_cap_label="none"
else
  event_cap_label="${max_events_per_interval} per interval"
fi
log "Event cap: ${event_cap_label}; EvMotionSeg downsample_rate: ${downsample_rate}"
log "EvMotionSeg terms: data=1, smooth=6000, label=60000, max_labels=24"

log "Checking Python dependencies."
"${python_bin}" - <<'PY' >>"${ev_out}/logs/dependency_check.log" 2>&1
import sys
sys.path.insert(0, "/home/DanielH/Optical_Flow/depthanyevent")
import cv2
import h5py
import hdf5plugin
import numpy
import yaml
from dataset.dsec_dataset.sbt.eventslicer import EventSlicer
print("dependency check ok")
PY

log "Building EvMotionSeg portable standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" \
  >"${ev_out}/logs/build_standalone.log" 2>&1

log "Preparing full-event EvMotionSeg text inputs from DSEC events and EMoFlow flow."
"${python_bin}" EvMotionSeg/tools/prepare_dsec_emoflow_for_evmotionseg.py \
  --sequence-dir "${sequence_dir}" \
  --flow-dir "${flow_dir}" \
  --output-dir "${ev_out}" \
  --max-events-per-interval "${max_events_per_interval}" \
  --interval-s 0.1 \
  --preview-stride 4 \
  >"${ev_out}/logs/prepare.log" 2>&1

num_intervals="$("${python_bin}" - "${ev_out}/evmotionseg_input_summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["num_intervals"])
PY
)"

log "Prepared ${num_intervals} intervals. Starting EvMotionSeg."
"${build_dir}/motion_segmentation_standalone" \
  --data_file_path "${ev_out}" \
  --interval 0.1 \
  --width 640 \
  --height 480 \
  --downsample_rate "${downsample_rate}" \
  --fx 569.7632987676102 \
  --fy 569.7632987676102 \
  --data_term 1 \
  --smooth_term 6000 \
  --label_term 60000 \
  --GraphCutIteration 10 \
  --MotionSegIteration 4 \
  --max_labels 24 \
  --num_intervals "${num_intervals}" \
  --imo_background_mode background_fit \
  >"${ev_out}/logs/evmotionseg.log" 2>&1

log "Generating qualitative overlay contact sheet, video, and summary."
"${python_bin}" - "${ev_out}" <<'PY' >"${ev_out}/logs/qualitative.log" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = Path(sys.argv[1])
preview_dir = base / "event_preview"
mask_dir = base / "results_imo"
out_dir = base / "qualitative"
out_dir.mkdir(exist_ok=True)

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
small_font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()


def moving_mask(mask_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path).convert("RGB"))
    return np.any(arr < 200, axis=2)


def make_overlay(idx: int) -> tuple[Image.Image, float] | None:
    preview_path = preview_dir / f"{idx:06d}.png"
    mask_path = mask_dir / f"{idx}.png"
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


all_mask_files = sorted(mask_dir.glob("*.png"), key=lambda path: int(path.stem))
all_fracs = [float(moving_mask(path).mean()) for path in all_mask_files]
overlays: list[tuple[int, Image.Image]] = []
for preview_path in sorted(preview_dir.glob("*.png"), key=lambda path: int(path.stem)):
    item = make_overlay(int(preview_path.stem))
    if item is not None:
        overlays.append((int(preview_path.stem), item[0]))

contact_sheet = out_dir / "dsec_evmotionseg_emoflow_all_events_contact_sheet.png"
video_path = out_dir / "dsec_evmotionseg_emoflow_all_events_stride4.mp4"

if overlays:
    sample_count = min(24, len(overlays))
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
    sheet.save(contact_sheet)

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (tile_w, tile_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")
    for _, image in overlays:
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()

arr = np.array(all_fracs, dtype=np.float64)
summary = {
    "run_dir": str(base),
    "num_masks": int(len(all_fracs)),
    "num_preview_overlays": int(len(overlays)),
    "moving_fraction_mean": float(np.mean(arr)) if arr.size else None,
    "moving_fraction_median": float(np.median(arr)) if arr.size else None,
    "moving_fraction_p90": float(np.percentile(arr, 90)) if arr.size else None,
    "moving_fraction_max": float(np.max(arr)) if arr.size else None,
    "nonzero_imo_frames": int(np.count_nonzero(arr > 0)) if arr.size else 0,
    "frames_over_1pct": int(np.count_nonzero(arr > 0.01)) if arr.size else 0,
    "frames_over_5pct": int(np.count_nonzero(arr > 0.05)) if arr.size else 0,
    "top_frames": [
        [int(all_mask_files[i].stem), float(arr[i])]
        for i in np.argsort(arr)[-10:][::-1]
    ]
    if arr.size
    else [],
    "contact_sheet": str(contact_sheet),
    "video": str(video_path),
}
(out_dir / "qualitative_summary_emoflow_all_events.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY

log "Full-event EvMotionSeg run complete."
