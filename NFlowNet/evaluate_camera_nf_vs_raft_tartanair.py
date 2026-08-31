#!/usr/bin/env python3
"""Compare Camera_NF variants and RAFT-small on TartanAir normal-flow GT."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.transforms.functional import convert_image_dtype

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Camera_NF import (  # noqa: E402
    GRID_STEP,
    PATCH_RADIUS,
    compute_normal_flow_affine,
    compute_normal_flow_baseline,
    compute_normal_flow_patch,
    compute_normal_flow_patch_affine,
    preprocess,
    select_strongest_gradient_points,
)


PAIR_RE = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)_normal_scalar\.npy$")


@dataclass
class MetricAccumulator:
    tested: int = 0
    valid_count: int = 0
    scalar_sq_sum: float = 0.0
    vector_epe_sum: float = 0.0
    scalar_abs_sum: float = 0.0
    frame_count: int = 0

    def update(
        self,
        pred_scalar: np.ndarray,
        pred_vector: np.ndarray,
        gt_scalar: np.ndarray,
        gt_vector: np.ndarray,
        test_mask: np.ndarray,
        method_valid: np.ndarray,
    ) -> None:
        valid_mask = test_mask & method_valid
        self.tested += int(test_mask.sum())
        self.frame_count += 1

        if not np.any(valid_mask):
            return

        scalar_error = pred_scalar[valid_mask] - gt_scalar[valid_mask]
        vector_error = pred_vector[valid_mask] - gt_vector[valid_mask]
        vector_epe = np.linalg.norm(vector_error, axis=1)

        self.valid_count += int(valid_mask.sum())
        self.scalar_sq_sum += float(np.sum(scalar_error * scalar_error))
        self.scalar_abs_sum += float(np.sum(np.abs(scalar_error)))
        self.vector_epe_sum += float(np.sum(vector_epe))

    def as_dict(self) -> dict[str, float | int]:
        valid_fraction = (
            self.valid_count / self.tested
            if self.tested > 0
            else float("nan")
        )

        if self.valid_count == 0:
            return {
                "tested_points": self.tested,
                "valid_points": 0,
                "frames": self.frame_count,
                "valid_fraction": float(valid_fraction),
                "rmse": float("nan"),
                "mae_scalar": float("nan"),
                "aepe": float("nan"),
            }

        return {
            "tested_points": self.tested,
            "valid_points": self.valid_count,
            "frames": self.frame_count,
            "valid_fraction": float(valid_fraction),
            "rmse": float(np.sqrt(self.scalar_sq_sum / self.valid_count)),
            "mae_scalar": float(self.scalar_abs_sum / self.valid_count),
            "aepe": float(self.vector_epe_sum / self.valid_count),
        }


@dataclass
class MethodMetrics:
    selected: MetricAccumulator = field(default_factory=MetricAccumulator)
    seconds: float = 0.0

    def as_dict(self, pairs: int) -> dict[str, dict[str, float | int | dict[str, float]]]:
        metrics = self.selected.as_dict()
        metrics["timing"] = {
            "seconds": float(self.seconds),
            "ms_per_pair": float(1000.0 * self.seconds / pairs),
            "pairs_per_second": float(pairs / self.seconds) if self.seconds > 0 else float("nan"),
            "ms_per_valid_point": (
                float(1000.0 * self.seconds / self.selected.valid_count)
                if self.selected.valid_count > 0
                else float("nan")
            ),
        }
        return {"selected": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Camera_NF deterministic normal flow and RAFT-small against "
            "TartanAir normal-flow GT."
        )
    )
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=REPO_ROOT / "Datasets/TartanAir/Office",
        help="TartanAir trajectory root containing image_left/ and normal_flow_gt/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/camera_nf_variants_vs_raft_small_office_metrics.json",
        help="Metrics JSON output path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/camera_nf_variants_vs_raft_small_office_metrics.csv",
        help="Flat metrics CSV output path.",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=max(GRID_STEP * 4, 80),
        help="Select one strongest GT/baseline-valid gradient point per grid cell.",
    )
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=PATCH_RADIUS,
        help="Patch radius for patch-based Camera_NF variants.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="RAFT batch size.",
    )
    parser.add_argument(
        "--num-flow-updates",
        type=int,
        default=12,
        help="RAFT recurrent update count.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Debug limit; 0 evaluates the whole sequence.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for RAFT.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly initialized RAFT-small instead of pretrained weights.",
    )
    return parser.parse_args()


def image_path(image_dir: Path, frame_id: str) -> Path:
    return image_dir / f"{frame_id}_left.png"


def load_pair_metadata(sequence_root: Path, max_pairs: int) -> list[tuple[str, str, Path]]:
    scalar_dir = sequence_root / "normal_flow_gt" / "scalar"
    scalar_paths = sorted(scalar_dir.glob("*_normal_scalar.npy"))
    if max_pairs > 0:
        scalar_paths = scalar_paths[:max_pairs]

    pairs: list[tuple[str, str, Path]] = []
    for scalar_path in scalar_paths:
        match = PAIR_RE.match(scalar_path.name)
        if match is None:
            continue
        pairs.append((match.group("first"), match.group("second"), scalar_path))

    if not pairs:
        raise FileNotFoundError(f"No normal-flow scalar GT found in {scalar_dir}")
    return pairs


def read_preprocessed_pair(
    first_image_path: Path,
    second_image_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    first_bgr = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    second_bgr = cv2.imread(str(second_image_path), cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise FileNotFoundError(first_image_path)
    if second_bgr is None:
        raise FileNotFoundError(second_image_path)

    first_gray = preprocess(first_bgr)
    second_gray = preprocess(second_bgr)

    return first_gray, second_gray


def load_gt(sequence_root: Path, first: str, second: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"{first}_{second}"
    gt_scalar = np.load(
        sequence_root / "normal_flow_gt" / "scalar" / f"{stem}_normal_scalar.npy"
    ).astype(np.float32)
    gradient_dir = np.load(
        sequence_root / "normal_flow_gt" / "gradient_dir" / f"{stem}_gradient_dir.npy"
    ).astype(np.float32)
    valid_mask = cv2.imread(
        str(sequence_root / "normal_flow_gt" / "valid_mask" / f"{stem}_valid.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if valid_mask is None:
        raise FileNotFoundError(
            sequence_root / "normal_flow_gt" / "valid_mask" / f"{stem}_valid.png"
        )
    gt_valid = valid_mask > 0
    gt_vector = (gt_scalar[..., None] * gradient_dir).astype(np.float32)
    return gt_scalar, gradient_dir, gt_valid, gt_vector


def vector_and_scalar(
    vx: np.ndarray,
    vy: np.ndarray,
    gradient_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_vector = np.stack((vx, vy), axis=-1).astype(np.float32)
    pred_scalar = np.sum(pred_vector * gradient_dir, axis=-1)
    return pred_vector, pred_scalar


def mask_from_points(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y, x in points:
        mask[y, x] = True
    return mask


def prepare_raft_batch(
    image_dir: Path,
    pairs: list[tuple[str, str, Path]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch1 = []
    batch2 = []
    for first, second, _scalar_path in pairs:
        im1 = convert_image_dtype(
            read_image(str(image_path(image_dir, first)), mode=ImageReadMode.RGB),
            torch.float32,
        )
        im2 = convert_image_dtype(
            read_image(str(image_path(image_dir, second)), mode=ImageReadMode.RGB),
            torch.float32,
        )
        batch1.append(im1)
        batch2.append(im2)

    return torch.stack(batch1).to(device), torch.stack(batch2).to(device)


def main() -> None:
    args = parse_args()
    sequence_root = args.sequence_root.expanduser().resolve()
    image_dir = sequence_root / "image_left"
    pairs = load_pair_metadata(sequence_root, args.max_pairs)

    device = torch.device(args.device)
    weights = None if args.no_pretrained else Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms() if weights is not None else None

    method_metrics = {
        "baseline": MethodMetrics(),
        "affine": MethodMetrics(),
        "patch": MethodMetrics(),
        "patch_affine": MethodMetrics(),
        "raft_small": MethodMetrics(),
    }
    raft_seconds = 0.0
    raft_forward_seconds = 0.0

    gt_cache: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    selected_counts = []
    affine_brightness = []
    patch_affine_brightness = []

    for index, (first, second, _scalar_path) in enumerate(pairs, start=1):
        gt_scalar, gradient_dir, gt_valid, gt_vector = load_gt(sequence_root, first, second)
        t0 = time.perf_counter()
        first_gray, second_gray = read_preprocessed_pair(
            image_path(image_dir, first),
            image_path(image_dir, second),
        )
        image_read_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        (
            vx,
            vy,
            _magnitude,
            baseline_valid,
            Ix,
            Iy,
            _It,
        ) = compute_normal_flow_baseline(
            first_gray,
            first_gray,
            second_gray,
            0.0,
            0.0,
            1.0,
        )
        method_metrics["baseline"].seconds += image_read_seconds + (
            time.perf_counter() - t0
        )
        baseline_vector, baseline_scalar = vector_and_scalar(vx, vy, gradient_dir)

        grad_strength = np.sqrt(Ix**2 + Iy**2)
        selected_points = select_strongest_gradient_points(
            gt_valid & baseline_valid,
            grad_strength,
            step=args.grid_step,
        )
        test_mask = mask_from_points(gt_valid.shape, selected_points)
        selected_counts.append(len(selected_points))

        method_metrics["baseline"].selected.update(
            baseline_scalar,
            baseline_vector,
            gt_scalar,
            gt_vector,
            test_mask,
            baseline_valid,
        )

        t0 = time.perf_counter()
        affine_result = compute_normal_flow_affine(
            first_gray,
            first_gray,
            second_gray,
            0.0,
            0.0,
            1.0,
        )
        method_metrics["affine"].seconds += image_read_seconds + (
            time.perf_counter() - t0
        )
        vx, vy, _magnitude, affine_valid, _Ix, _Iy, _It, brightness = affine_result
        affine_brightness.append(brightness)
        affine_vector, affine_scalar = vector_and_scalar(vx, vy, gradient_dir)
        method_metrics["affine"].selected.update(
            affine_scalar,
            affine_vector,
            gt_scalar,
            gt_vector,
            test_mask,
            affine_valid,
        )

        t0 = time.perf_counter()
        patch_result = compute_normal_flow_patch(
            first_gray,
            first_gray,
            second_gray,
            0.0,
            0.0,
            1.0,
            locations=selected_points,
            patch_radius=args.patch_radius,
        )
        method_metrics["patch"].seconds += image_read_seconds + (
            time.perf_counter() - t0
        )
        vx, vy, _magnitude, patch_valid, _confidence, _photo, _Ix, _Iy = patch_result
        patch_vector, patch_scalar = vector_and_scalar(vx, vy, gradient_dir)
        method_metrics["patch"].selected.update(
            patch_scalar,
            patch_vector,
            gt_scalar,
            gt_vector,
            test_mask,
            patch_valid,
        )

        t0 = time.perf_counter()
        patch_affine_result = compute_normal_flow_patch_affine(
            first_gray,
            first_gray,
            second_gray,
            0.0,
            0.0,
            1.0,
            locations=selected_points,
            patch_radius=args.patch_radius,
        )
        method_metrics["patch_affine"].seconds += image_read_seconds + (
            time.perf_counter() - t0
        )
        (
            vx,
            vy,
            _magnitude,
            patch_affine_valid,
            _confidence,
            _photo,
            _Ix,
            _Iy,
            brightness,
        ) = patch_affine_result
        patch_affine_brightness.append(brightness)
        patch_affine_vector, patch_affine_scalar = vector_and_scalar(vx, vy, gradient_dir)
        method_metrics["patch_affine"].selected.update(
            patch_affine_scalar,
            patch_affine_vector,
            gt_scalar,
            gt_vector,
            test_mask,
            patch_affine_valid,
        )

        gt_cache.append((gt_scalar, gradient_dir, gt_valid, gt_vector, test_mask))

        if index % 50 == 0 or index == len(pairs):
            print(
                f"Camera_NF variants evaluated {index}/{len(pairs)} pairs",
                flush=True,
            )

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            end = min(start + args.batch_size, len(pairs))
            batch_pairs = pairs[start:end]
            t0 = time.perf_counter()
            images1, images2 = prepare_raft_batch(image_dir, batch_pairs, device)
            if transforms is not None:
                images1, images2 = transforms(images1, images2)
            else:
                images1 = images1 * 2.0 - 1.0
                images2 = images2 * 2.0 - 1.0

            forward_t0 = time.perf_counter()
            flow = raft(images1, images2, num_flow_updates=args.num_flow_updates)[-1]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            raft_forward_seconds += time.perf_counter() - forward_t0
            flow_np = flow.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            raft_seconds += time.perf_counter() - t0

            for offset in range(end - start):
                gt_scalar, gradient_dir, _gt_valid, gt_vector, test_mask = gt_cache[
                    start + offset
                ]
                pred_scalar = np.sum(flow_np[offset] * gradient_dir, axis=-1)
                pred_vector = (pred_scalar[..., None] * gradient_dir).astype(np.float32)
                raft_valid = np.isfinite(flow_np[offset]).all(axis=-1)

                method_metrics["raft_small"].selected.update(
                    pred_scalar,
                    pred_vector,
                    gt_scalar,
                    gt_vector,
                    test_mask,
                    raft_valid,
                )

            if end % (args.batch_size * 10) == 0 or end == len(pairs):
                print(f"RAFT-small evaluated {end}/{len(pairs)} pairs", flush=True)

    method_metrics["raft_small"].seconds = raft_seconds

    result = {
        "sequence_root": str(sequence_root),
        "pairs": len(pairs),
        "selection": {
            "grid_step": args.grid_step,
            "patch_radius": args.patch_radius,
            "mean_points_per_pair": float(np.mean(selected_counts)),
            "min_points_per_pair": int(np.min(selected_counts)),
            "max_points_per_pair": int(np.max(selected_counts)),
        },
        "device": str(device),
        "raft": {
            "weights": None if weights is None else weights.name,
            "num_flow_updates": args.num_flow_updates,
        },
        "metrics_note": (
            "Metrics are computed at identical selected GT-valid and "
            "baseline-gradient-valid points for every method. "
            "RMSE is scalar normal-flow RMSE in pixels/frame. "
            "AEPE is vector endpoint error after comparing predicted normal-flow "
            "vectors to TartanAir normal-flow GT vectors."
        ),
        "timing_note": (
            "Camera_NF timing covers image read, preprocessing, and each "
            "method's flow computation. RAFT-small timing covers image read, "
            "TorchVision transforms, model forward, and CPU transfer; "
            "forward_only covers the synchronized model forward pass."
        ),
        "affine_brightness_debug": {
            "affine_mean": {
                key: float(np.mean([item[key] for item in affine_brightness]))
                for key in ("a_prev", "b_prev", "a_next", "b_next")
            },
            "patch_affine_mean": {
                key: float(np.mean([item[key] for item in patch_affine_brightness]))
                for key in ("a_prev", "b_prev", "a_next", "b_next")
            },
        },
        "methods": {
            name: metrics.as_dict(len(pairs))
            for name, metrics in method_metrics.items()
        },
    }
    result["methods"]["raft_small"]["selected"]["forward_only_timing"] = {
        "seconds": float(raft_forward_seconds),
        "ms_per_pair": float(1000.0 * raft_forward_seconds / len(pairs)),
        "pairs_per_second": (
            float(len(pairs) / raft_forward_seconds)
            if raft_forward_seconds > 0
            else float("nan")
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "method,tested_points,valid_points,valid_fraction,rmse,mae_scalar,aepe,seconds,ms_per_pair,pairs_per_second,ms_per_valid_point"
    ]
    for name, method_result in result["methods"].items():
        selected = method_result["selected"]
        timing = selected["timing"]
        rows.append(
            ",".join(
                [
                    name,
                    str(selected["tested_points"]),
                    str(selected["valid_points"]),
                    f"{selected['valid_fraction']:.12g}",
                    f"{selected['rmse']:.12g}",
                    f"{selected['mae_scalar']:.12g}",
                    f"{selected['aepe']:.12g}",
                    f"{timing['seconds']:.12g}",
                    f"{timing['ms_per_pair']:.12g}",
                    f"{timing['pairs_per_second']:.12g}",
                    f"{timing['ms_per_valid_point']:.12g}",
                ]
            )
        )
    args.summary_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
