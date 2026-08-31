#!/usr/bin/env python3
"""Evaluate RAFT-small with the DSO-weighted patch NF protocol."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.transforms.functional import convert_image_dtype

from Camera_NF import (
    compute_image_derivatives,
    preprocess,
    select_strongest_gradient_points,
)
from evaluate_dso_patch_nf import flow_pairs, load_flow_gt, normal_gt_from_flow


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=REPO_ROOT / "Datasets/TartanAir/Office",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results/dso_weighted_patch_nf",
    )
    parser.add_argument("--dev-pairs", type=int, default=5)
    parser.add_argument("--eval-pairs", type=int, default=80)
    parser.add_argument("--grid-step", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-flow-updates", type=int, default=12)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def image_path(sequence_root: Path, frame_id: str) -> Path:
    return sequence_root / "image_left" / f"{frame_id}_left.png"


def prepare_batch(
    sequence_root: Path,
    pairs: list[tuple[str, str, Path]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch1 = []
    batch2 = []
    for first, second, _flow_path in pairs:
        im1 = convert_image_dtype(
            read_image(str(image_path(sequence_root, first)), mode=ImageReadMode.RGB),
            torch.float32,
        )
        im2 = convert_image_dtype(
            read_image(str(image_path(sequence_root, second)), mode=ImageReadMode.RGB),
            torch.float32,
        )
        batch1.append(im1)
        batch2.append(im2)
    return torch.stack(batch1).to(device), torch.stack(batch2).to(device)


def load_normal_reference(
    sequence_root: Path,
    first: str,
    second: str,
    grid_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    first_bgr = cv2.imread(str(image_path(sequence_root, first)), cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise FileNotFoundError(image_path(sequence_root, first))
    I0 = preprocess(first_bgr)
    ix, iy = compute_image_derivatives(I0)
    grad_mag = np.sqrt(ix * ix + iy * iy)
    flow, flow_valid = load_flow_gt(sequence_root, first, second)
    gt_scalar, gt_vector, gradient_valid = normal_gt_from_flow(flow, ix, iy)
    eval_valid = flow_valid & gradient_valid
    locations = select_strongest_gradient_points(eval_valid, grad_mag, step=grid_step)
    nx = np.zeros_like(ix, dtype=np.float32)
    ny = np.zeros_like(iy, dtype=np.float32)
    nx[gradient_valid] = ix[gradient_valid] / grad_mag[gradient_valid]
    ny[gradient_valid] = iy[gradient_valid] / grad_mag[gradient_valid]
    return gt_scalar, gt_vector, nx, ny, locations


def summarize(errors: list[float], epes: list[float], seconds: float, pairs: int) -> dict:
    err = np.asarray(errors, dtype=np.float64)
    epe = np.asarray(epes, dtype=np.float64)
    return {
        "valid_points": int(err.size),
        "mean_abs_normal_error": float(np.mean(err)),
        "median_abs_normal_error": float(np.median(err)),
        "p90_abs_normal_error": float(np.percentile(err, 90)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "aepe": float(np.mean(epe)),
        "bad_0p5": float(np.mean(err > 0.5)),
        "bad_1": float(np.mean(err > 1.0)),
        "bad_2": float(np.mean(err > 2.0)),
        "seconds": float(seconds),
        "ms_per_frame": float(1000.0 * seconds / pairs),
    }


def main() -> None:
    args = parse_args()
    sequence_root = args.sequence_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_pairs = flow_pairs(sequence_root)
    eval_pairs = all_pairs[args.dev_pairs : args.dev_pairs + args.eval_pairs]
    if not eval_pairs:
        raise ValueError("No evaluation pairs selected")

    device = torch.device(args.device)
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=True).to(device).eval()

    references = [
        load_normal_reference(sequence_root, first, second, args.grid_step)
        for first, second, _flow_path in eval_pairs
    ]

    errors: list[float] = []
    epes: list[float] = []
    rows = []
    total_seconds = 0.0
    forward_seconds = 0.0

    with torch.no_grad():
        for start in range(0, len(eval_pairs), args.batch_size):
            end = min(start + args.batch_size, len(eval_pairs))
            batch_pairs = eval_pairs[start:end]

            t0 = time.perf_counter()
            images1, images2 = prepare_batch(sequence_root, batch_pairs, device)
            images1, images2 = transforms(images1, images2)

            forward_t0 = time.perf_counter()
            flow_pred = model(
                images1,
                images2,
                num_flow_updates=args.num_flow_updates,
            )[-1]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - forward_t0

            flow_np = flow_pred.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            total_seconds += time.perf_counter() - t0

            for offset, (first, _second, _flow_path) in enumerate(batch_pairs):
                gt_scalar, gt_vector, nx, ny, locations = references[start + offset]
                flow = flow_np[offset]

                pred_scalar = flow[..., 0] * nx + flow[..., 1] * ny
                pred_vector = np.stack((pred_scalar * nx, pred_scalar * ny), axis=-1)
                for y, x in locations:
                    abs_error = float(abs(pred_scalar[y, x] - gt_scalar[y, x]))
                    epe = float(np.linalg.norm(pred_vector[y, x] - gt_vector[y, x]))
                    errors.append(abs_error)
                    epes.append(epe)
                    rows.append(
                        {
                            "frame": first,
                            "x": x,
                            "y": y,
                            "method": "raft_small",
                            "absolute_normal_error": abs_error,
                            "normal_epe": epe,
                            "gt_normal_displacement": float(gt_scalar[y, x]),
                            "estimated_normal_displacement": float(pred_scalar[y, x]),
                        }
                    )
            print(f"RAFT-small evaluated {end}/{len(eval_pairs)} pairs", flush=True)

    summary = {
        "sequence_root": str(sequence_root),
        "dev_pairs": args.dev_pairs,
        "eval_pairs": len(eval_pairs),
        "grid_step": args.grid_step,
        "device": str(device),
        "weights": weights.name,
        "num_flow_updates": args.num_flow_updates,
        "summary": summarize(errors, epes, total_seconds, len(eval_pairs)),
        "forward_only": {
            "seconds": float(forward_seconds),
            "ms_per_frame": float(1000.0 * forward_seconds / len(eval_pairs)),
        },
    }

    (output_root / "raft_small_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "raft_small_raw_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
