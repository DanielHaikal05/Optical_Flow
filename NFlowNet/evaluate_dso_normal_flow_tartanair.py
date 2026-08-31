#!/usr/bin/env python3
"""Run DSO on TartanAir and evaluate sparse DSO-derived normal flow.

DSO estimates sparse inverse-depth map points and camera poses. This script
projects each DSO point from its host frame into the next frame, projects that
full displacement onto the GT image-gradient direction, and evaluates the
result against TartanAir normal-flow GT sampled at the same sparse locations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_RE = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)_normal_scalar\.npy$")


@dataclass
class MetricAccumulator:
    tested_points: int = 0
    valid_points: int = 0
    scalar_sq_sum: float = 0.0
    scalar_abs_sum: float = 0.0
    epe_sum: float = 0.0

    def update(
        self,
        pred_scalar: float,
        pred_vector: np.ndarray,
        gt_scalar: float,
        gt_vector: np.ndarray,
    ) -> None:
        self.tested_points += 1
        if not (
            math.isfinite(pred_scalar)
            and np.isfinite(pred_vector).all()
            and math.isfinite(gt_scalar)
            and np.isfinite(gt_vector).all()
        ):
            return

        scalar_error = pred_scalar - gt_scalar
        vector_error = pred_vector - gt_vector
        self.valid_points += 1
        self.scalar_sq_sum += scalar_error * scalar_error
        self.scalar_abs_sum += abs(scalar_error)
        self.epe_sum += float(np.linalg.norm(vector_error))

    def as_dict(self) -> dict[str, float | int]:
        if self.valid_points == 0:
            return {
                "tested_points": self.tested_points,
                "valid_points": 0,
                "valid_fraction": 0.0,
                "rmse": float("nan"),
                "mae_scalar": float("nan"),
                "aepe": float("nan"),
            }

        return {
            "tested_points": self.tested_points,
            "valid_points": self.valid_points,
            "valid_fraction": self.valid_points / self.tested_points
            if self.tested_points
            else float("nan"),
            "rmse": math.sqrt(self.scalar_sq_sum / self.valid_points),
            "mae_scalar": self.scalar_abs_sum / self.valid_points,
            "aepe": self.epe_sum / self.valid_points,
        }


@dataclass
class Pose:
    incoming_id: int
    internal_id: int
    timestamp: float
    valid: bool
    fx: float
    fy: float
    cx: float
    cy: float
    cam_to_world: np.ndarray


@dataclass
class EvaluatedPoint:
    event_id: int
    final: int
    point_group: str
    output: str
    host_id: int
    target_id: int
    u: float
    v: float
    projected_u: float
    projected_v: float
    full_flow_x: float
    full_flow_y: float
    normal_scalar: float
    normal_flow_x: float
    normal_flow_y: float
    gt_scalar: float
    gt_flow_x: float
    gt_flow_y: float
    scalar_error: float
    epe: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSO on TartanAir Office and evaluate sparse normal flow."
    )
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=REPO_ROOT / "Datasets/TartanAir/Office",
        help="TartanAir sequence root containing image_left/ and normal_flow_gt/.",
    )
    parser.add_argument(
        "--dso-binary",
        type=Path,
        default=REPO_ROOT / "dso/build/bin/dso_dataset_headless",
        help="Headless DSO binary.",
    )
    parser.add_argument(
        "--dso-input-root",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/dso_office_input",
        help="Generated DSO input folder with symlinked images/times/calibration.",
    )
    parser.add_argument(
        "--dso-output-root",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/dso_office",
        help="DSO output folder.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/dso_office_metrics.json",
        help="Aggregate metrics JSON output.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/dso_office_metrics.csv",
        help="Aggregate metrics CSV output.",
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/dso_office_sparse_normal_flow_points.csv",
        help="Sparse normal-flow point predictions CSV output.",
    )
    parser.add_argument(
        "--frame-metrics-csv",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/dso_office_frame_metrics.csv",
        help="Per-frame sparse metrics CSV output.",
    )
    parser.add_argument(
        "--run-dso",
        action="store_true",
        help="Run DSO before evaluating exported poses/points.",
    )
    parser.add_argument(
        "--skip-input-prepare",
        action="store_true",
        help="Do not regenerate DSO input symlinks/times/calibration.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit DSO/evaluation to the first N frames; 0 uses all available frames.",
    )
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Synthetic timestamp rate for DSO times.txt.",
    )
    parser.add_argument(
        "--dso-arg",
        action="append",
        default=[],
        help=(
            "Extra raw DSO argument, repeatable. Defaults are "
            "preset=0 mode=2 quiet=1 nolog=1 nomt=1."
        ),
    )
    return parser.parse_args()


def image_paths(sequence_root: Path) -> list[Path]:
    return sorted((sequence_root / "image_left").glob("*_left.png"))


def prepare_dso_input(
    sequence_root: Path,
    input_root: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fps: float,
    max_frames: int,
) -> Path:
    images = image_paths(sequence_root)
    if max_frames > 0:
        images = images[:max_frames]
    if not images:
        raise FileNotFoundError(sequence_root / "image_left")

    input_root.mkdir(parents=True, exist_ok=True)
    image_dir = input_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(images):
        dst = image_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            dst.unlink()
        os.symlink(src.resolve(), dst)

    times_path = input_root / "times.txt"
    with times_path.open("w", encoding="utf-8") as f:
        for idx, _src in enumerate(images):
            f.write(f"{idx} {idx / fps:.9f} 1.0\n")

    calib_path = input_root / "camera.txt"
    with calib_path.open("w", encoding="utf-8") as f:
        f.write(f"Pinhole {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g} 0\n")
        f.write("640 480\n")
        f.write("none\n")
        f.write("640 480\n")

    return image_dir


def run_dso(args: argparse.Namespace, dso_image_dir: Path) -> float:
    args.dso_output_root.mkdir(parents=True, exist_ok=True)
    dso_args = args.dso_arg or ["preset=0", "mode=2", "quiet=1", "nolog=1", "nomt=1"]
    cmd = [
        str(args.dso_binary),
        f"files={dso_image_dir}",
        f"calib={args.dso_input_root / 'camera.txt'}",
        f"nfoutput={args.dso_output_root}",
        *dso_args,
    ]
    if args.max_frames > 0:
        cmd.append(f"end={args.max_frames}")

    started = time.perf_counter()
    log_path = args.dso_output_root / "dso_stdout.log"
    env = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"DSO failed with exit code {proc.returncode}; see {log_path}"
        )
    return elapsed


def load_poses(path: Path) -> dict[int, Pose]:
    poses: dict[int, Pose] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = np.eye(4, dtype=np.float64)
            values = [float(row[f"t{r}{c}"]) for r in range(3) for c in range(4)]
            T[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
            pose = Pose(
                incoming_id=int(row["incoming_id"]),
                internal_id=int(row["internal_id"]),
                timestamp=float(row["timestamp"]),
                valid=bool(int(row["pose_valid"])),
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                cam_to_world=T,
            )
            poses[pose.incoming_id] = pose
    return poses


def load_pair_ids(sequence_root: Path) -> set[tuple[int, int]]:
    pairs = set()
    scalar_dir = sequence_root / "normal_flow_gt" / "scalar"
    for path in sorted(scalar_dir.glob("*_normal_scalar.npy")):
        match = PAIR_RE.match(path.name)
        if match is None:
            continue
        pairs.add((int(match.group("first")), int(match.group("second"))))
    return pairs


def bilinear(array: np.ndarray, x: float, y: float) -> np.ndarray | float:
    h, w = array.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        raise ValueError("sample outside bilinear domain")

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0

    top = (1.0 - dx) * array[y0, x0] + dx * array[y0, x0 + 1]
    bottom = (1.0 - dx) * array[y0 + 1, x0] + dx * array[y0 + 1, x0 + 1]
    return (1.0 - dy) * top + dy * bottom


@dataclass
class GtCache:
    sequence_root: Path
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    def get(self, first: int, second: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (first, second)
        if key not in self.cache:
            stem = f"{first:06d}_{second:06d}"
            scalar = np.load(
                self.sequence_root
                / "normal_flow_gt"
                / "scalar"
                / f"{stem}_normal_scalar.npy"
            ).astype(np.float32)
            gradient_dir = np.load(
                self.sequence_root
                / "normal_flow_gt"
                / "gradient_dir"
                / f"{stem}_gradient_dir.npy"
            ).astype(np.float32)
            valid_image = cv2.imread(
                str(
                    self.sequence_root
                    / "normal_flow_gt"
                    / "valid_mask"
                    / f"{stem}_valid.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            if valid_image is None:
                raise FileNotFoundError(f"{stem}_valid.png")
            self.cache[key] = (scalar, gradient_dir, valid_image > 0)
        return self.cache[key]


def project_point(row: dict[str, str], host_pose: Pose, target_pose: Pose) -> tuple[float, float, float, float]:
    u = float(row["u"])
    v = float(row["v"])
    idepth = float(row["idepth"])
    if idepth <= 0 or not math.isfinite(idepth):
        raise ValueError("invalid inverse depth")

    z = 1.0 / idepth
    x = (u - float(row["cx"])) / float(row["fx"]) * z
    y = (v - float(row["cy"])) / float(row["fy"]) * z
    host_point = np.array([x, y, z, 1.0], dtype=np.float64)

    world_point = host_pose.cam_to_world @ host_point
    target_world_to_cam = np.linalg.inv(target_pose.cam_to_world)
    target_point = target_world_to_cam @ world_point
    if target_point[2] <= 0:
        raise ValueError("point projects behind target camera")

    projected_u = target_pose.fx * target_point[0] / target_point[2] + target_pose.cx
    projected_v = target_pose.fy * target_point[1] / target_point[2] + target_pose.cy
    return float(projected_u), float(projected_v), u, v


def output_labels(row: dict[str, str]) -> list[str]:
    group = row["point_group"]
    final = int(row["final"])
    final_name = "final" if final else "nonfinal"
    return [
        "all",
        f"all_{final_name}",
        group,
        f"{group}_{final_name}",
    ]


def evaluate(args: argparse.Namespace, dso_seconds: float | None) -> dict:
    poses_path = args.dso_output_root / "poses.csv"
    points_path = args.dso_output_root / "keyframe_points.csv"
    if not poses_path.exists():
        raise FileNotFoundError(poses_path)
    if not points_path.exists():
        raise FileNotFoundError(points_path)

    poses = load_poses(poses_path)
    pair_ids = load_pair_ids(args.sequence_root)
    gt_cache = GtCache(args.sequence_root)

    metrics: dict[str, MetricAccumulator] = {}
    frame_metrics: dict[tuple[int, str], MetricAccumulator] = {}
    latest_points: dict[tuple[str, int, int, int], EvaluatedPoint] = {}

    evaluated_rows = 0
    skipped = {
        "no_pair": 0,
        "missing_pose": 0,
        "invalid_projection": 0,
        "outside_image": 0,
        "invalid_gt": 0,
    }

    eval_started = time.perf_counter()
    with points_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            host_id = int(row["host_incoming_id"])
            target_id = host_id + 1
            if args.max_frames > 0 and target_id >= args.max_frames:
                continue
            if (host_id, target_id) not in pair_ids:
                skipped["no_pair"] += 1
                continue
            if host_id not in poses or target_id not in poses:
                skipped["missing_pose"] += 1
                continue
            if not poses[host_id].valid or not poses[target_id].valid:
                skipped["missing_pose"] += 1
                continue

            try:
                projected_u, projected_v, u, v = project_point(
                    row, poses[host_id], poses[target_id]
                )
            except ValueError:
                skipped["invalid_projection"] += 1
                continue

            scalar_gt, gradient_dir, valid_mask = gt_cache.get(host_id, target_id)
            h, w = scalar_gt.shape
            if u < 0 or v < 0 or u >= w - 1 or v >= h - 1:
                skipped["outside_image"] += 1
                continue
            nearest_x = int(round(u))
            nearest_y = int(round(v))
            if nearest_x < 0 or nearest_y < 0 or nearest_x >= w or nearest_y >= h:
                skipped["outside_image"] += 1
                continue
            if not valid_mask[nearest_y, nearest_x]:
                skipped["invalid_gt"] += 1
                continue

            gt_scalar = float(bilinear(scalar_gt, u, v))
            grad = np.asarray(bilinear(gradient_dir, u, v), dtype=np.float64)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm <= 1e-8 or not math.isfinite(grad_norm):
                skipped["invalid_gt"] += 1
                continue
            grad = grad / grad_norm

            full_flow = np.array([projected_u - u, projected_v - v], dtype=np.float64)
            normal_scalar = float(np.dot(full_flow, grad))
            normal_flow = normal_scalar * grad
            gt_flow = gt_scalar * grad
            scalar_error = normal_scalar - gt_scalar
            epe = float(np.linalg.norm(normal_flow - gt_flow))

            evaluated = EvaluatedPoint(
                event_id=int(row["event_id"]),
                final=int(row["final"]),
                point_group=row["point_group"],
                output="",
                host_id=host_id,
                target_id=target_id,
                u=u,
                v=v,
                projected_u=projected_u,
                projected_v=projected_v,
                full_flow_x=float(full_flow[0]),
                full_flow_y=float(full_flow[1]),
                normal_scalar=normal_scalar,
                normal_flow_x=float(normal_flow[0]),
                normal_flow_y=float(normal_flow[1]),
                gt_scalar=gt_scalar,
                gt_flow_x=float(gt_flow[0]),
                gt_flow_y=float(gt_flow[1]),
                scalar_error=scalar_error,
                epe=epe,
            )

            for label in output_labels(row):
                labeled_point = EvaluatedPoint(**{**evaluated.__dict__, "output": label})
                metrics.setdefault(label, MetricAccumulator()).update(
                    normal_scalar, normal_flow, gt_scalar, gt_flow
                )
                frame_metrics.setdefault((host_id, label), MetricAccumulator()).update(
                    normal_scalar, normal_flow, gt_scalar, gt_flow
                )

                dedupe_key = (label, host_id, nearest_x, nearest_y)
                old = latest_points.get(dedupe_key)
                if old is None or labeled_point.event_id >= old.event_id:
                    latest_points[dedupe_key] = labeled_point

            evaluated_rows += 1

    evaluation_seconds = time.perf_counter() - eval_started

    latest_metrics: dict[str, MetricAccumulator] = {}
    for (label, _host_id, _x, _y), point in latest_points.items():
        latest_metrics.setdefault(label, MetricAccumulator()).update(
            point.normal_scalar,
            np.array([point.normal_flow_x, point.normal_flow_y], dtype=np.float64),
            point.gt_scalar,
            np.array([point.gt_flow_x, point.gt_flow_y], dtype=np.float64),
        )

    args.points_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.points_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(EvaluatedPoint.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in latest_points.values():
            writer.writerow(point.__dict__)

    args.frame_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.frame_metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "host_id",
            "target_id",
            "output",
            "tested_points",
            "valid_points",
            "valid_fraction",
            "rmse",
            "mae_scalar",
            "aepe",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (host_id, label), accumulator in sorted(frame_metrics.items()):
            row = accumulator.as_dict()
            writer.writerow(
                {
                    "host_id": host_id,
                    "target_id": host_id + 1,
                    "output": label,
                    **row,
                }
            )

    aggregate = {
        label: accumulator.as_dict()
        for label, accumulator in sorted(metrics.items())
    }
    latest_aggregate = {
        f"{label}_latest_unique": accumulator.as_dict()
        for label, accumulator in sorted(latest_metrics.items())
    }

    num_pairs = len(pair_ids)
    num_input_frames = len(image_paths(args.sequence_root))
    if args.max_frames > 0:
        num_input_frames = min(num_input_frames, args.max_frames)
    timing = {
        "dso_run_seconds": dso_seconds,
        "dso_ms_per_input_frame": (
            1000.0 * dso_seconds / max(1, num_input_frames)
            if dso_seconds is not None
            else None
        ),
        "evaluation_seconds": evaluation_seconds,
        "evaluation_ms_per_evaluated_point": (
            1000.0 * evaluation_seconds / evaluated_rows if evaluated_rows else None
        ),
    }

    result = {
        "sequence_root": str(args.sequence_root.resolve()),
        "dso_output_root": str(args.dso_output_root.resolve()),
        "num_gt_pairs": num_pairs,
        "camera_assumption": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "width": 640,
            "height": 480,
            "note": (
                "TartanAir Office calibration was not present in the local folder; "
                "this run uses the standard 640x480, 90-degree-FOV pinhole assumption."
            ),
        },
        "metric_note": (
            "DSO outputs are sparse. RMSE is scalar normal-flow RMSE in pixels/frame; "
            "AEPE is endpoint error between DSO-derived sparse normal-flow vectors and "
            "GT normal-flow vectors sampled at the same DSO point locations."
        ),
        "counts": {
            "input_frames": num_input_frames,
            "evaluated_snapshot_rows": evaluated_rows,
            "latest_unique_labeled_sparse_points": len(latest_points),
            "skipped": skipped,
        },
        "timing": timing,
        "snapshot_metrics": aggregate,
        "latest_unique_metrics": latest_aggregate,
        "outputs": {
            "poses_csv": str(poses_path.resolve()),
            "dso_points_csv": str(points_path.resolve()),
            "sparse_normal_flow_points_csv": str(args.points_csv.resolve()),
            "frame_metrics_csv": str(args.frame_metrics_csv.resolve()),
        },
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "metric_set",
            "output",
            "tested_points",
            "valid_points",
            "valid_fraction",
            "rmse",
            "mae_scalar",
            "aepe",
            "dso_run_seconds",
            "dso_ms_per_input_frame",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in aggregate.items():
            writer.writerow(
                {
                    "metric_set": "snapshots",
                    "output": label,
                    **row,
                    "dso_run_seconds": dso_seconds,
                    "dso_ms_per_input_frame": timing["dso_ms_per_input_frame"],
                }
            )
        for label, row in latest_aggregate.items():
            writer.writerow(
                {
                    "metric_set": "latest_unique",
                    "output": label,
                    **row,
                    "dso_run_seconds": dso_seconds,
                    "dso_ms_per_input_frame": timing["dso_ms_per_input_frame"],
                }
            )

    return result


def main() -> None:
    args = parse_args()
    args.sequence_root = args.sequence_root.expanduser().resolve()
    args.dso_binary = args.dso_binary.expanduser().resolve()
    args.dso_input_root = args.dso_input_root.expanduser().resolve()
    args.dso_output_root = args.dso_output_root.expanduser().resolve()

    if args.skip_input_prepare:
        dso_image_dir = args.dso_input_root / "images"
    else:
        dso_image_dir = prepare_dso_input(
            args.sequence_root,
            args.dso_input_root,
            args.fx,
            args.fy,
            args.cx,
            args.cy,
            args.fps,
            args.max_frames,
        )

    dso_seconds = run_dso(args, dso_image_dir) if args.run_dso else None
    result = evaluate(args, dso_seconds)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
