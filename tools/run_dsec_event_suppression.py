#!/usr/bin/env python3
"""Run Event Suppression masks on prepared DSEC smart-sampled streams."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EVSUP_ROOT = ROOT / "event_suppression"
if str(EVSUP_ROOT) not in sys.path:
    sys.path.insert(0, str(EVSUP_ROOT))

from evsup.config import load_config  # noqa: E402
from evsup.models.model_hydra import HydraEVNet  # noqa: E402
from evsup.utils.utils import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        action="append",
        required=True,
        help="DSEC sequence directory. Repeat for multiple sequences.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "analysis" / f"event_suppression_dsec_20k_density_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EVSUP_ROOT / "checkpoints" / "event_suppression_checkpoints" / "model_epoch_49.pth",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EVSUP_ROOT / "evsup" / "configs" / "validate_evimo.json",
    )
    parser.add_argument("--python-bin", type=Path, default=ROOT / "E-MoFlow" / ".venv" / "bin" / "python")
    parser.add_argument("--max-events-per-interval", type=int, default=20_000)
    parser.add_argument("--sampling-mode", choices=("density", "global", "grid"), default="density")
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--density-power", type=float, default=0.75)
    parser.add_argument("--density-offset", type=float, default=1.0)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--preview-stride", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-intervals", type=int, default=0, help="Debug limit. 0 means the full sequence.")
    parser.add_argument("--reuse-samples", action="store_true")
    return parser.parse_args()


def run_sampler(args: argparse.Namespace, sequence_dir: Path, sample_dir: Path, log_path: Path) -> None:
    if args.reuse_samples and (sample_dir / "dsec_veckm_input_summary.json").exists():
        return

    command = [
        str(args.python_bin),
        str(ROOT / "EvMotionSeg" / "tools" / "prepare_dsec_veckm_input.py"),
        "--sequence-dir",
        str(sequence_dir),
        "--output-dir",
        str(sample_dir),
        "--interval-s",
        str(args.interval_s),
        "--preview-stride",
        str(args.preview_stride),
        "--overwrite",
        "--sampling-mode",
        args.sampling_mode,
    ]
    if args.max_intervals:
        command += ["--num-intervals", str(args.max_intervals)]
    if args.sampling_mode == "density":
        command += [
            "--max-events-per-interval",
            str(args.max_events_per_interval),
            "--grid-cols",
            str(args.grid_cols),
            "--grid-rows",
            str(args.grid_rows),
            "--density-power",
            str(args.density_power),
            "--density-offset",
            str(args.density_offset),
        ]
    elif args.sampling_mode == "global":
        command += ["--max-events-per-interval", str(args.max_events_per_interval)]
    else:
        per_cell = max(1, args.max_events_per_interval // max(1, args.grid_cols * args.grid_rows))
        command += [
            "--max-events-per-interval",
            "0",
            "--grid-cols",
            str(args.grid_cols),
            "--grid-rows",
            str(args.grid_rows),
            "--max-events-per-grid-cell",
            str(per_cell),
        ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def normalized_voxel(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    height: int,
    width: int,
    num_bins: int,
) -> torch.Tensor:
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if t.size < 2:
        return torch.from_numpy(voxel)

    t0 = float(t[0])
    denom = float(t[-1] - t[0])
    if denom <= 0.0:
        return torch.from_numpy(voxel)

    t_norm = (num_bins - 1) * (t.astype(np.float64) - t0) / denom
    x0 = x.astype(np.int64, copy=False)
    y0 = y.astype(np.int64, copy=False)
    b0 = np.floor(t_norm).astype(np.int64)
    values = 2.0 * p.astype(np.float32, copy=False) - 1.0

    for b in (b0, b0 + 1):
        valid = (b >= 0) & (b < num_bins) & (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
        weights = values * (1.0 - np.abs(b.astype(np.float64) - t_norm)).astype(np.float32)
        np.add.at(voxel, (b[valid], y0[valid], x0[valid]), weights[valid])

    nonzero = voxel != 0
    if np.any(nonzero):
        values_nz = voxel[nonzero]
        mean = float(values_nz.mean())
        std = float(values_nz.std())
        voxel[nonzero] = (values_nz - mean) / std if std > 0.0 else values_nz - mean
    return torch.from_numpy(voxel)


def render_interval_events(width: int, height: int, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> Image.Image:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    pos = p > 0
    image[y[~pos], x[~pos]] = (255, 96, 32)
    image[y[pos], x[pos]] = (32, 64, 255)
    return Image.fromarray(image)


def mask_overlay(base: Image.Image, mask: np.ndarray, title: str, font: ImageFont.ImageFont) -> Image.Image:
    image = np.asarray(base.convert("RGB")).astype(np.float32)
    image[mask] = image[mask] * 0.25 + np.array([0, 220, 80], dtype=np.float32) * 0.75
    out = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 28), fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)
    return out


def write_contact_sheet(images: list[Image.Image], path: Path, cols: int = 4) -> None:
    if not images:
        return
    thumb_w = 320
    thumb_h = int(round(images[0].height * (thumb_w / images[0].width)))
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))
    for idx, image in enumerate(images):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        sheet.paste(thumb, ((idx % cols) * thumb_w, (idx // cols) * thumb_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def load_event_suppressor(config_path: Path, checkpoint: Path, device: torch.device) -> tuple[HydraEVNet, dict[str, Any]]:
    config = load_config(config_path)
    model = HydraEVNet(
        kwargs=config["model"].copy(),
        num_bins=config["data"].get("voxel_bins", 2),
        final_w_scale_flow=config["custom"].get("final_w_scale_flow", 0.01),
        current_flow_sup=config["custom"].get("current_flow_sup", False),
        current_flow_scaling=config["loader"].get("event_dt_ms", 50),
    ).to(device)
    model = load_model(model, device, str(checkpoint))
    model.eval()
    return model, config


def run_inference(
    args: argparse.Namespace,
    model: HydraEVNet,
    config: dict[str, Any],
    sequence_dir: Path,
    sample_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    summary = json.loads((sample_dir / "dsec_veckm_input_summary.json").read_text(encoding="utf-8"))
    intervals = summary["intervals"]
    if args.max_intervals:
        intervals = intervals[: args.max_intervals]

    width = int(summary["width"])
    height = int(summary["height"])
    num_bins = int(config["data"].get("voxel_bins", 2))
    device = torch.device(args.device)

    xy = np.load(sample_dir / "dataset_events_xy.npy", mmap_mode="r")
    timestamps = np.load(sample_dir / "dataset_events_t.npy", mmap_mode="r")
    polarity = np.load(sample_dir / "dataset_events_p.npy", mmap_mode="r")

    masks_dir = output_dir / "masks"
    prob_dir = output_dir / "probability"
    overlay_dir = output_dir / "overlays"
    for directory in (masks_dir, prob_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    csv_rows: list[dict[str, Any]] = []
    preview_images: list[Image.Image] = []

    model.reset_states()
    with torch.no_grad():
        for item in intervals:
            interval_index = int(item["interval_index"])
            start_us = int(round(float(item["start_s"]) * 1_000_000.0))
            end_us = int(round(float(item["end_s"]) * 1_000_000.0))
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))

            x = np.asarray(xy[left:right, 0])
            y = np.asarray(xy[left:right, 1])
            p = np.asarray(polarity[left:right])
            t = np.asarray(timestamps[left:right])

            voxel = normalized_voxel(x, y, p, t, height, width, num_bins).to(device)
            dt = torch.tensor([[float(args.interval_s)]], dtype=torch.float32, device=device)
            output = model(voxel.unsqueeze(0), dt)
            logits = output["mask"][-1].squeeze(0).squeeze(0)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            mask = prob > args.threshold

            Image.fromarray((prob * 255.0).clip(0, 255).astype(np.uint8)).save(
                prob_dir / f"{interval_index:06d}.png"
            )
            Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(masks_dir / f"{interval_index:06d}.png")

            event_count = int(right - left)
            frac = float(mask.mean())
            csv_rows.append(
                {
                    "interval_index": interval_index,
                    "start_s": item["start_s"],
                    "end_s": item["end_s"],
                    "events": event_count,
                    "dynamic_fraction": frac,
                    "max_probability": float(prob.max()) if prob.size else 0.0,
                    "mean_probability": float(prob.mean()) if prob.size else 0.0,
                }
            )

            if args.preview_stride > 0 and len(csv_rows) % args.preview_stride == 1:
                preview_path = sample_dir / "event_preview" / f"{interval_index:06d}.png"
                base = Image.open(preview_path) if preview_path.exists() else render_interval_events(width, height, x, y, p)
                title = f"{sequence_dir.name} frame {interval_index:06d} mask {frac * 100:.2f}%"
                overlay = mask_overlay(base, mask, title, font)
                overlay.save(overlay_dir / f"{interval_index:06d}.png")
                preview_images.append(overlay)

    with (output_dir / "mask_fraction_by_interval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    top_overlays: list[Image.Image] = []
    for row in sorted(csv_rows, key=lambda item: item["dynamic_fraction"], reverse=True)[:24]:
        interval_index = int(row["interval_index"])
        start_us = int(round(float(row["start_s"]) * 1_000_000.0))
        end_us = int(round(float(row["end_s"]) * 1_000_000.0))
        left = int(np.searchsorted(timestamps, start_us, side="left"))
        right = int(np.searchsorted(timestamps, end_us, side="left"))
        x = np.asarray(xy[left:right, 0])
        y = np.asarray(xy[left:right, 1])
        p = np.asarray(polarity[left:right])
        preview_path = sample_dir / "event_preview" / f"{interval_index:06d}.png"
        base = Image.open(preview_path) if preview_path.exists() else render_interval_events(width, height, x, y, p)
        mask = np.asarray(Image.open(masks_dir / f"{interval_index:06d}.png").convert("L")) < 128
        title = f"{sequence_dir.name} frame {interval_index:06d} mask {row['dynamic_fraction'] * 100:.2f}%"
        overlay = mask_overlay(base, mask, title, font)
        overlay.save(overlay_dir / f"top_{interval_index:06d}.png")
        top_overlays.append(overlay)

    write_contact_sheet(preview_images[:24], output_dir / "contact_sheet_stride.png")
    write_contact_sheet(top_overlays, output_dir / "contact_sheet_top_mask_fraction.png")

    fractions = np.asarray([row["dynamic_fraction"] for row in csv_rows], dtype=np.float64)
    result = {
        "sequence": sequence_dir.name,
        "sequence_dir": str(sequence_dir),
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "representation": "normalized signed-polarity two-bin voxel grid",
        "threshold": args.threshold,
        "num_intervals": len(csv_rows),
        "total_sampled_events": int(sum(row["events"] for row in csv_rows)),
        "mean_dynamic_fraction": float(fractions.mean()) if fractions.size else 0.0,
        "median_dynamic_fraction": float(np.median(fractions)) if fractions.size else 0.0,
        "max_dynamic_fraction": float(fractions.max()) if fractions.size else 0.0,
        "nonzero_mask_frames": int(np.count_nonzero(fractions > 0.0)),
        "frames_over_1pct": int(np.count_nonzero(fractions > 0.01)),
        "frames_over_5pct": int(np.count_nonzero(fractions > 0.05)),
        "top_dynamic_fraction_frames": [
            [int(csv_rows[i]["interval_index"]), float(fractions[i])]
            for i in np.argsort(fractions)[::-1][:10]
        ],
        "sampler_summary": {key: value for key, value in summary.items() if key != "intervals"},
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, config = load_event_suppressor(args.config, args.checkpoint, device)

    all_results = []
    for sequence_dir in args.sequence_dir:
        sequence_dir = sequence_dir.resolve()
        label = sequence_dir.name
        density_slug = str(args.density_power).replace(".", "p")
        sample_dir = args.output_root / label / f"{label}_{args.sampling_mode}{args.max_events_per_interval // 1000}k_p{density_slug}"
        output_dir = args.output_root / label / "event_suppression"
        run_sampler(args, sequence_dir, sample_dir, output_dir / "logs" / "prepare_dsec_veckm_input.log")
        all_results.append(run_inference(args, model, config, sequence_dir, sample_dir, output_dir))

    combined = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(args.output_root),
        "results": all_results,
    }
    (args.output_root / "summary.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
