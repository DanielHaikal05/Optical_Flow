#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

export PYTHONPATH="/home/DanielH/Optical_Flow/VecKM_flow:${PYTHONPATH:-}"

max_events_per_interval="${EVMOTIONSEG_MAX_EVENTS_PER_INTERVAL:-20000}"
downsample_rate="${EVMOTIONSEG_DOWNSAMPLE_RATE:-1}"
smooth_term="${EVMOTIONSEG_SMOOTH_TERM:-6000}"
label_term="${EVMOTIONSEG_LABEL_TERM:-60000}"
max_labels="${EVMOTIONSEG_MAX_LABELS:-24}"
preview_stride="${EVMOTIONSEG_PREVIEW_STRIDE:-4}"
imo_background_mode="${EVMOTIONSEG_IMO_BACKGROUND_MODE:-background_fit}"
sampling_mode="${EVMOTIONSEG_SAMPLING_MODE:-global}"
sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/interlaken_00_c}"
sequence_label="${DSEC_SEQUENCE_LABEL:-$(basename "${sequence_dir}")}"
grid_cols="${EVMOTIONSEG_GRID_COLS:-8}"
grid_rows="${EVMOTIONSEG_GRID_ROWS:-6}"
max_events_per_grid_cell="${EVMOTIONSEG_MAX_EVENTS_PER_GRID_CELL:-1000}"
density_power="${EVMOTIONSEG_DENSITY_POWER:-0.75}"
density_offset="${EVMOTIONSEG_DENSITY_OFFSET:-1.0}"
if [[ "${sampling_mode}" == "grid" ]]; then
  if (( max_events_per_grid_cell >= 1000 && max_events_per_grid_cell % 1000 == 0 )); then
    grid_cell_label="$((max_events_per_grid_cell / 1000))k"
  else
    grid_cell_label="${max_events_per_grid_cell}"
  fi
  event_cap_label="grid${grid_cols}x${grid_rows}_cell${grid_cell_label}"
elif [[ "${sampling_mode}" == "density" ]]; then
  event_cap_k="$((max_events_per_interval / 1000))"
  density_power_slug="${density_power//./p}"
  event_cap_label="density${event_cap_k}k_p${density_power_slug}"
else
  event_cap_k="$((max_events_per_interval / 1000))"
  event_cap_label="${event_cap_k}k"
fi
label_term_slug="${label_term//./p}"
smooth_term_slug="${smooth_term//./p}"
term_suffix=""
if [[ "${label_term}" != "60000" ]]; then
  term_suffix="${term_suffix}_label${label_term_slug}"
fi
if [[ "${smooth_term}" != "6000" ]]; then
  term_suffix="${term_suffix}_smooth${smooth_term_slug}"
fi
run_prefix="${EVMOTIONSEG_RUN_PREFIX:-dsec_${sequence_label}_veckm_${event_cap_label}_down${downsample_rate}_bgfit${term_suffix}}"
run_name="${run_prefix}_$(date +%Y%m%d_%H%M%S)"
run_root="VecKM_flow/outputs/${run_name}"
sample_dir="${run_root}/${sequence_label}_${event_cap_label}"
veckm_out_root="${run_root}/veckm"
veckm_pred_dir="${veckm_out_root}/$(basename "${sample_dir}")"
ev_out="EvMotionSeg/data/${run_name}"
latest_link="EvMotionSeg/data/${run_prefix}_latest"
latest_path_file="EvMotionSeg/data/${run_prefix}_latest_path.txt"
veckm_latest_link="VecKM_flow/outputs/${run_prefix}_latest"
veckm_latest_path_file="VecKM_flow/outputs/${run_prefix}_latest_path.txt"
status_log="${ev_out}/logs/status.log"
python_bin="/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python"
build_dir="/tmp/dsec_evmotionseg_veckm${event_cap_label}_build"

mkdir -p "${ev_out}/logs" "${run_root}"
ln -sfn "$(basename "${ev_out}")" "${latest_link}"
printf '%s\n' "${ev_out}" > "${latest_path_file}"
ln -sfn "$(basename "${run_root}")" "${veckm_latest_link}"
printf '%s\n' "${run_root}" > "${veckm_latest_path_file}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Started DSEC VecKM -> EvMotionSeg ${event_cap_label} run on $(hostname)."
log "Output directory: ${ev_out}"
log "VecKM run root: ${run_root}"
log "Sequence: ${sequence_dir}"
log "Sequence label: ${sequence_label}"
if [[ "${sampling_mode}" == "grid" ]]; then
  log "Sampling mode: grid ${grid_cols}x${grid_rows}; max ${max_events_per_grid_cell} events/cell/interval; nominal cap $((grid_cols * grid_rows * max_events_per_grid_cell)) per 0.1 s interval; EvMotionSeg downsample_rate: ${downsample_rate}"
elif [[ "${sampling_mode}" == "density" ]]; then
  log "Sampling mode: density ${grid_cols}x${grid_rows}; cap ${max_events_per_interval} events/interval; density_power=${density_power}; density_offset=${density_offset}; EvMotionSeg downsample_rate: ${downsample_rate}"
else
  log "Sampling mode: global; event cap: ${max_events_per_interval} per 0.1 s interval; EvMotionSeg downsample_rate: ${downsample_rate}"
fi
log "Normal-flow source: VecKM DSEC checkpoint, not EMoFlow."
log "EvMotionSeg terms: data=1, smooth=${smooth_term}, label=${label_term}, max_labels=${max_labels}, imo_background_mode=${imo_background_mode}"

log "Checking Python dependencies."
"${python_bin}" - <<'PY' >>"${ev_out}/logs/dependency_check.log" 2>&1
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

log "Building EvMotionSeg portable standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" \
  >"${ev_out}/logs/build_standalone.log" 2>&1

sample_args=(
  --sequence-dir "${sequence_dir}" \
  --output-dir "${sample_dir}" \
  --interval-s 0.1 \
  --preview-stride "${preview_stride}" \
  --overwrite
)
if [[ "${sampling_mode}" == "grid" ]]; then
  sample_args+=(
    --sampling-mode grid
    --max-events-per-interval 0
    --grid-cols "${grid_cols}"
    --grid-rows "${grid_rows}"
    --max-events-per-grid-cell "${max_events_per_grid_cell}"
  )
elif [[ "${sampling_mode}" == "density" ]]; then
  sample_args+=(
    --sampling-mode density
    --max-events-per-interval "${max_events_per_interval}"
    --grid-cols "${grid_cols}"
    --grid-rows "${grid_rows}"
    --density-power "${density_power}"
    --density-offset "${density_offset}"
  )
else
  sample_args+=(
    --sampling-mode global
    --max-events-per-interval "${max_events_per_interval}"
  )
fi

log "Sampling DSEC events into VecKM input arrays."
"${python_bin}" EvMotionSeg/tools/prepare_dsec_veckm_input.py "${sample_args[@]}" \
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
log "Prepared ${num_intervals} intervals with ${written_events} rectified sampled events."
log "Rectified event camera: width=${ev_width}, height=${ev_height}, fx=${ev_fx}, fy=${ev_fy}"

log "Running VecKM normal-flow inference."
"${python_bin}" VecKM_flow/run_veckm_evimo2_sliding.py \
  --data-root "$(dirname "${sample_dir}")" \
  --sequences "$(basename "${sample_dir}")" \
  --output-dir "${veckm_out_root}" \
  --training-set DSEC \
  --ensemble 3 \
  --save-undistorted \
  >"${ev_out}/logs/veckm.log" 2>&1

log "Rendering VecKM flow masks."
"${python_bin}" EvMotionSeg/tools/render_veckm_flow_masks.py \
  "${sample_dir}" \
  "${veckm_pred_dir}" \
  "${ev_out}/veckm_flow" \
  --flow-file dataset_pred_flow_DSEC.npy \
  --uncertainty-file dataset_angle_vars_flow_DSEC.npy \
  --timestamp-scale 1e-6 \
  --interval-s 0.1 \
  --num-intervals "${num_intervals}" \
  --uncertainty-threshold 0.3 \
  >"${ev_out}/logs/veckm_flow_masks.log" 2>&1

log "Converting VecKM predictions into EvMotionSeg text inputs."
"${python_bin}" EvMotionSeg/tools/prepare_drone_sequence.py text \
  "${sample_dir}" \
  "${veckm_pred_dir}" \
  "${ev_out}" \
  --flow-file dataset_pred_flow_DSEC.npy \
  --timestamp-scale 1e-6 \
  --chunk-size 250000 \
  >"${ev_out}/logs/evmotionseg_text.log" 2>&1
rm -rf "${ev_out}/event_preview"
cp -a "${sample_dir}/event_preview" "${ev_out}/event_preview"

log "Starting EvMotionSeg."
"${build_dir}/motion_segmentation_standalone" \
  --data_file_path "${ev_out}" \
  --interval 0.1 \
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

log "Generating qualitative overlay contact sheet, video, and summary."
"${python_bin}" - "${ev_out}" "${event_cap_label}" <<'PY' >"${ev_out}/logs/qualitative.log" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = Path(sys.argv[1])
event_cap_label = sys.argv[2]
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


def make_imo_overlay(idx: int) -> tuple[Image.Image, float] | None:
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


def make_raw_overlay(idx: int) -> tuple[Image.Image, int] | None:
    preview_path = preview_dir / f"{idx:06d}.png"
    mask_path = raw_results_dir / f"{idx}.png"
    if not preview_path.exists() or not mask_path.exists():
        return None
    preview = np.asarray(Image.open(preview_path).convert("RGB")).astype(np.float32)
    labels = np.asarray(Image.open(mask_path).convert("RGB"))
    foreground = np.any(labels < 245, axis=2)
    blended = preview.copy()
    blended[foreground] = preview[foreground] * 0.35 + labels[foreground].astype(np.float32) * 0.65
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    colors = np.unique(labels[foreground].reshape(-1, 3), axis=0) if np.any(foreground) else np.empty((0, 3))
    label_count = int(colors.shape[0])
    draw = ImageDraw.Draw(out)
    label = f"t={idx * 0.1:05.1f}s  frame {idx:03d}  labels {label_count}"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, bbox[2] + 20, bbox[3] + 18), fill=(0, 0, 0))
    draw.text((14, 11), label, fill=(255, 255, 255), font=font)
    return out, label_count


all_imo_mask_files = sorted(imo_results_dir.glob("*.png"), key=lambda path: int(path.stem))
all_fracs = [float(moving_mask(path).mean()) for path in all_imo_mask_files]
imo_overlays: list[tuple[int, Image.Image]] = []
raw_overlays: list[tuple[int, Image.Image]] = []
raw_label_counts: list[tuple[int, int]] = []
for preview_path in sorted(preview_dir.glob("*.png"), key=lambda path: int(path.stem)):
    item = make_imo_overlay(int(preview_path.stem))
    if item is not None:
        imo_overlays.append((int(preview_path.stem), item[0]))
    raw_item = make_raw_overlay(int(preview_path.stem))
    if raw_item is not None:
        raw_overlays.append((int(preview_path.stem), raw_item[0]))
        raw_label_counts.append((int(preview_path.stem), raw_item[1]))

contact_sheet = out_dir / f"dsec_evmotionseg_veckm_{event_cap_label}_imo_contact_sheet.png"
video_path = out_dir / f"dsec_evmotionseg_veckm_{event_cap_label}_imo_stride4.mp4"
raw_contact_sheet = out_dir / f"dsec_evmotionseg_veckm_{event_cap_label}_raw_contact_sheet.png"
raw_video_path = out_dir / f"dsec_evmotionseg_veckm_{event_cap_label}_raw_stride4.mp4"

def write_contact_sheet(overlays: list[tuple[int, Image.Image]], path: Path) -> None:
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
    sheet.save(path)

def write_video(overlays: list[tuple[int, Image.Image]], path: Path) -> None:
    tile_w, tile_h = overlays[0][1].size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (tile_w, tile_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for _, image in overlays:
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()

if imo_overlays:
    write_contact_sheet(imo_overlays, contact_sheet)
    write_video(imo_overlays, video_path)

if raw_overlays:
    write_contact_sheet(raw_overlays, raw_contact_sheet)
    write_video(raw_overlays, raw_video_path)

arr = np.array(all_fracs, dtype=np.float64)
raw_counts_arr = np.array([count for _, count in raw_label_counts], dtype=np.float64)
summary = {
    "run_dir": str(base),
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
    "top_frames": [
        [int(all_imo_mask_files[i].stem), float(arr[i])]
        for i in np.argsort(arr)[-10:][::-1]
    ]
    if arr.size
    else [],
    "contact_sheet": str(contact_sheet),
    "video": str(video_path),
    "raw_label_count_mean": float(np.mean(raw_counts_arr)) if raw_counts_arr.size else None,
    "raw_label_count_median": float(np.median(raw_counts_arr)) if raw_counts_arr.size else None,
    "raw_label_count_max": int(np.max(raw_counts_arr)) if raw_counts_arr.size else None,
    "raw_top_label_count_frames": sorted(raw_label_counts, key=lambda item: item[1], reverse=True)[:10],
    "raw_contact_sheet": str(raw_contact_sheet),
    "raw_video": str(raw_video_path),
}
(out_dir / f"qualitative_summary_veckm_{event_cap_label}.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY

log "DSEC VecKM -> EvMotionSeg ${event_cap_label} run complete."
