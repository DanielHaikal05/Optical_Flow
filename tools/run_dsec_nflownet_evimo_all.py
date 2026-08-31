#!/usr/bin/env python3
"""Run the downloaded NFlowNet/EVIMO model on DSEC all-event windows.

The checkpoint at ``model_weights.pth`` is an older 6-channel image-pair model
that predicts a dense scalar normal-flow image.  For DSEC we render every
0.1 s event interval as a 3-channel event image, run the model on consecutive
event-image pairs, lift the scalar prediction back to a 2D normal-flow vector
using the event-image gradient direction, and sample that dense vector at every
rectified event pixel before writing EvMotionSeg text inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import hdf5plugin  # noqa: F401 - needed for DSEC compressed HDF5
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEPTHANYEVENT_ROOT = ROOT / "depthanyevent"
if str(DEPTHANYEVENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHANYEVENT_ROOT))

from dataset.dsec_dataset.sbt.eventslicer import EventSlicer  # noqa: E402


class OldNFlowNet(nn.Module):
    """Architecture matching the downloaded 920 KiB ``model_weights.pth``."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 48, kernel_size=5, stride=2, padding=2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(48, 96, kernel_size=5, stride=2, padding=2), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(96, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_weights.pth")
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/dsec_evmotionseg_nflownet_evimo_build"))
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--start-interval", type=int, default=0)
    parser.add_argument("--num-intervals", type=int, default=0, help="0 means all intervals.")
    parser.add_argument("--preview-stride", type=int, default=4)
    parser.add_argument("--flow-scale", type=float, default=1.0)
    parser.add_argument("--min-gradient", type=float, default=0.01)
    parser.add_argument("--gradient-blur-sigma", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--downsample-rate", type=int, default=1)
    parser.add_argument("--smooth-term", type=float, default=6000.0)
    parser.add_argument("--label-term", type=float, default=60000.0)
    parser.add_argument("--max-labels", type=int, default=24)
    parser.add_argument("--imo-background-mode", default="background_fit")
    parser.add_argument("--skip-evmotionseg", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_existing_path(sequence_dir: Path, candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n  {checked}")


def resolve_dsec_paths(sequence_dir: Path) -> tuple[Path, Path, Path]:
    events_path = first_existing_path(
        sequence_dir,
        [
            sequence_dir / "Events" / "events.h5",
            sequence_dir / "event_data" / "left" / "events.h5",
            sequence_dir / "event_data" / "events.h5",
            sequence_dir / "events" / "left" / "events.h5",
            sequence_dir / "events" / "events.h5",
            sequence_dir / "events.h5",
        ],
        "DSEC events.h5",
    )
    rectify_path = first_existing_path(
        sequence_dir,
        [
            sequence_dir / "Events" / "rectify_map.h5",
            sequence_dir / "event_data" / "left" / "rectify_map.h5",
            sequence_dir / "event_data" / "rectify_map.h5",
            sequence_dir / "events" / "left" / "rectify_map.h5",
            sequence_dir / "events" / "rectify_map.h5",
            sequence_dir / "rectify_map.h5",
        ],
        "DSEC rectify_map.h5",
    )
    calibration_path = first_existing_path(
        sequence_dir,
        [
            sequence_dir / "Calibration" / "cam_to_cam.yaml",
            sequence_dir / "calibration" / "cam_to_cam.yaml",
            sequence_dir / "cam_to_cam.yaml",
        ],
        "DSEC cam_to_cam.yaml",
    )
    return events_path, rectify_path, calibration_path


def load_rectified_intrinsics(calibration_path: Path) -> dict[str, Any]:
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)
    cam = calibration["intrinsics"]["camRect0"]
    fx, fy, cx, cy = [float(value) for value in cam["camera_matrix"]]
    width, height = [int(value) for value in cam["resolution"]]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def load_rectify_map(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["rectify_map"][:]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
        shutil.rmtree(output_dir)
    for rel in ("logs", "event_preview", "nflownet/scalar", "nflownet/vector_preview", "results", "results_next"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def load_model(checkpoint: Path, device: torch.device) -> OldNFlowNet:
    model = OldNFlowNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def rectify_events(events: dict[str, np.ndarray], rectify_map: np.ndarray, intr: dict[str, Any]) -> dict[str, np.ndarray]:
    if events["t"].shape[0] == 0:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_u8 = np.empty(0, dtype=np.uint8)
        return {"x": empty_i64, "y": empty_i64, "t": empty_i64, "p": empty_u8}

    x_raw = events["x"].astype(np.int64)
    y_raw = events["y"].astype(np.int64)
    xy_rect = rectify_map[y_raw, x_raw]
    x_rect = np.rint(xy_rect[:, 0]).astype(np.int64)
    y_rect = np.rint(xy_rect[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(xy_rect[:, 0])
        & np.isfinite(xy_rect[:, 1])
        & (x_rect >= 0)
        & (x_rect < intr["width"])
        & (y_rect >= 0)
        & (y_rect < intr["height"])
    )
    return {
        "x": x_rect[valid],
        "y": y_rect[valid],
        "t": events["t"].astype(np.int64)[valid],
        "p": events["p"].astype(np.uint8)[valid],
    }


def render_event_rgb(width: int, height: int, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    image = np.ones((height, width, 3), dtype=np.float32)
    if x.size:
        pos = p > 0
        image[y[~pos], x[~pos]] = (1.0, 0.376, 0.125)
        image[y[pos], x[pos]] = (0.125, 0.251, 1.0)
    return image


def gradient_direction(image_rgb: np.ndarray, min_gradient: float, blur_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor((image_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    valid = mag >= min_gradient
    safe = np.maximum(mag, 1.0e-6)
    direction = np.stack((gx / safe, gy / safe), axis=-1).astype(np.float32)
    direction[~valid] = 0.0
    return direction, valid


@torch.no_grad()
def predict_vector(
    model: OldNFlowNet,
    frame0: np.ndarray,
    frame1: np.ndarray,
    device: torch.device,
    min_gradient: float,
    blur_sigma: float,
    flow_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = np.concatenate([frame0, frame1], axis=2).transpose(2, 0, 1)[None]
    tensor = torch.from_numpy(pair).to(device=device, dtype=torch.float32)
    scalar = model(tensor)[0, 0].detach().cpu().numpy().astype(np.float32) * flow_scale
    direction, valid = gradient_direction(frame0, min_gradient=min_gradient, blur_sigma=blur_sigma)
    vector = scalar[..., None] * direction
    vector[~valid] = 0.0
    return scalar, vector.astype(np.float32), valid


def write_event_preview(path: Path, frame: np.ndarray) -> None:
    image = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def write_vector_preview(path: Path, vector: np.ndarray, valid: np.ndarray) -> None:
    mag, angle = cv2.cartToPolar(vector[..., 0], vector[..., 1], angleInDegrees=True)
    vals = mag[valid]
    scale = float(np.percentile(vals, 99.0)) if vals.size else 1.0
    scale = max(scale, 1.0e-6)
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
    hsv[..., 2] = np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))


def append_interval_text(
    events_f,
    undist_f,
    flow_f,
    events: dict[str, np.ndarray],
    vector: np.ndarray,
    sequence_start_us: int,
    intr: dict[str, Any],
    chunk_size: int,
) -> int:
    written = 0
    total = int(events["t"].shape[0])
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        x = events["x"][start:end]
        y = events["y"][start:end]
        t = (events["t"][start:end].astype(np.float64) - sequence_start_us) * 1.0e-6
        p = events["p"][start:end]
        flow = vector[y, x]
        xn = (x.astype(np.float64) - intr["cx"]) / intr["fx"]
        yn = (y.astype(np.float64) - intr["cy"]) / intr["fy"]
        events_f.writelines(
            f"{tt:.9f} {int(xx)} {int(yy)} {int(pp)}\n"
            for tt, xx, yy, pp in zip(t, x, y, p)
        )
        undist_f.writelines(f"{float(xx):.9f} {float(yy):.9f}\n" for xx, yy in zip(xn, yn))
        flow_f.writelines(f"{float(ff[0]):.9f} {float(ff[1]):.9f}\n" for ff in flow)
        written += end - start
    return written


def run_evmotionseg(args: argparse.Namespace, intr: dict[str, Any], num_intervals: int) -> None:
    build_script = ROOT / "EvMotionSeg" / "tools" / "build_standalone_portable.sh"
    subprocess.run([str(build_script), str(args.build_dir)], check=True, stdout=(args.output_dir / "logs/build_standalone.log").open("w"), stderr=subprocess.STDOUT)
    binary = args.build_dir / "motion_segmentation_standalone"
    cmd = [
        str(binary),
        "--data_file_path",
        str(args.output_dir),
        "--interval",
        str(args.interval_s),
        "--width",
        str(intr["width"]),
        "--height",
        str(intr["height"]),
        "--downsample_rate",
        str(args.downsample_rate),
        "--fx",
        str(intr["fx"]),
        "--fy",
        str(intr["fy"]),
        "--data_term",
        "1",
        "--smooth_term",
        str(args.smooth_term),
        "--label_term",
        str(args.label_term),
        "--GraphCutIteration",
        "10",
        "--MotionSegIteration",
        "4",
        "--max_labels",
        str(args.max_labels),
        "--num_intervals",
        str(num_intervals),
        "--imo_background_mode",
        args.imo_background_mode,
    ]
    with (args.output_dir / "logs/evmotionseg.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)


def moving_mask(mask_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path).convert("RGB"))
    return np.any(arr < 200, axis=2)


def summarize_outputs(output_dir: Path) -> dict[str, Any]:
    mask_files = sorted((output_dir / "results_imo").glob("*.png"), key=lambda path: int(path.stem)) if (output_dir / "results_imo").exists() else []
    fractions = np.array([float(moving_mask(path).mean()) for path in mask_files], dtype=np.float64)
    summary = {
        "num_masks": int(len(mask_files)),
        "moving_fraction_mean": float(fractions.mean()) if fractions.size else None,
        "moving_fraction_median": float(np.median(fractions)) if fractions.size else None,
        "moving_fraction_p90": float(np.percentile(fractions, 90)) if fractions.size else None,
        "moving_fraction_max": float(fractions.max()) if fractions.size else None,
        "frames_over_1pct": int(np.count_nonzero(fractions > 0.01)) if fractions.size else 0,
        "frames_over_5pct": int(np.count_nonzero(fractions > 0.05)) if fractions.size else 0,
        "top_frames": [[int(mask_files[i].stem), float(fractions[i])] for i in np.argsort(fractions)[-10:][::-1]] if fractions.size else [],
    }
    return summary


def main() -> int:
    args = parse_args()
    sequence_dir = args.sequence_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()

    prepare_output_dir(args.output_dir, args.overwrite)
    events_path, rectify_path, calibration_path = resolve_dsec_paths(sequence_dir)
    intr = load_rectified_intrinsics(calibration_path)
    rectify_map = load_rectify_map(rectify_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    interval_summaries: list[dict[str, Any]] = []
    with h5py.File(events_path, "r") as event_file:
        slicer = EventSlicer(event_file)
        sequence_start_us = int(slicer.get_start_time_us())
        final_us = int(slicer.get_final_time_us())
        interval_us = int(round(args.interval_s * 1_000_000.0))
        available_intervals = max(0, math.ceil((final_us - sequence_start_us) / interval_us))
        start_interval = max(0, args.start_interval)
        end_interval = available_intervals if args.num_intervals <= 0 else min(available_intervals, start_interval + args.num_intervals)
        selected = list(range(start_interval, end_interval))
        if not selected:
            raise ValueError("No intervals selected")

        def load_interval(interval_index: int) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
            start_us = sequence_start_us + interval_index * interval_us
            end_us = min(start_us + interval_us, final_us)
            raw = slicer.get_events(start_us, end_us)
            rect = rectify_events(raw, rectify_map, intr)
            frame = render_event_rgb(intr["width"], intr["height"], rect["x"], rect["y"], rect["p"])
            return rect, frame, int(raw["t"].shape[0])

        current_events, current_frame, current_raw_count = load_interval(selected[0])
        events_txt = args.output_dir / "events.txt"
        undist_txt = args.output_dir / "undistorted_normalized_xy.txt"
        flow_txt = args.output_dir / "flow_xy.txt"
        with events_txt.open("w", encoding="utf-8") as events_f, undist_txt.open("w", encoding="utf-8") as undist_f, flow_txt.open("w", encoding="utf-8") as flow_f:
            for pos, interval_index in enumerate(tqdm(selected, desc=f"NFlowNet DSEC {sequence_dir.name}")):
                if pos + 1 < len(selected):
                    next_events, next_frame, next_raw_count = load_interval(selected[pos + 1])
                else:
                    next_events, next_frame, next_raw_count = current_events, current_frame, current_raw_count

                scalar, vector, valid_gradient = predict_vector(
                    model,
                    current_frame,
                    next_frame,
                    device,
                    min_gradient=args.min_gradient,
                    blur_sigma=args.gradient_blur_sigma,
                    flow_scale=args.flow_scale,
                )
                written = append_interval_text(
                    events_f,
                    undist_f,
                    flow_f,
                    current_events,
                    vector,
                    sequence_start_us,
                    intr,
                    args.chunk_size,
                )
                if args.preview_stride > 0 and (interval_index - start_interval) % args.preview_stride == 0:
                    write_event_preview(args.output_dir / "event_preview" / f"{interval_index:06d}.png", current_frame)
                    np.save(args.output_dir / "nflownet/scalar" / f"{interval_index:06d}.npy", scalar)
                    write_vector_preview(args.output_dir / "nflownet/vector_preview" / f"{interval_index:06d}.png", vector, valid_gradient)

                interval_summaries.append(
                    {
                        "interval_index": interval_index,
                        "raw_events": current_raw_count,
                        "valid_rectified_events": int(current_events["t"].shape[0]),
                        "written_events": written,
                        "gradient_valid_fraction": float(valid_gradient.mean()),
                        "scalar_mean": float(np.mean(scalar)),
                        "scalar_std": float(np.std(scalar)),
                        "vector_norm_mean": float(np.linalg.norm(vector.reshape(-1, 2), axis=1).mean()),
                    }
                )
                current_events, current_frame, current_raw_count = next_events, next_frame, next_raw_count

    written_counts = [item["written_events"] for item in interval_summaries]
    summary = {
        "sequence": sequence_dir.name,
        "sequence_dir": str(sequence_dir),
        "events_path": str(events_path),
        "rectify_path": str(rectify_path),
        "calibration_path": str(calibration_path),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "model": "OldNFlowNet downloaded EVIMO checkpoint",
        "sampling": "none",
        "interval_s": args.interval_s,
        "start_interval": args.start_interval,
        "num_intervals": len(interval_summaries),
        "width": intr["width"],
        "height": intr["height"],
        "fx": intr["fx"],
        "fy": intr["fy"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "written_events": int(sum(written_counts)),
        "mean_written_events_per_interval": float(np.mean(written_counts)) if written_counts else 0.0,
        "flow_scale": args.flow_scale,
        "min_gradient": args.min_gradient,
        "gradient_blur_sigma": args.gradient_blur_sigma,
        "skip_evmotionseg": bool(args.skip_evmotionseg),
        "intervals": interval_summaries,
    }
    (args.output_dir / "nflownet_input_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.skip_evmotionseg:
        run_evmotionseg(args, intr, len(interval_summaries))
        summary["evmotionseg"] = summarize_outputs(args.output_dir)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "intervals"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
