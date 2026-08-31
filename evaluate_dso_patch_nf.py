#!/usr/bin/env python3
"""Evaluate DSO-weighted direct patch normal flow on TartanAir."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Camera_NF import (  # noqa: E402
    DSO_SIGMA,
    DSO_WEIGHT_LAMBDA,
    GRAD_THRESHOLD,
    ORIENTATION_GAMMA,
    PATCH_RADIUS,
    compute_image_derivatives,
    compute_normal_flow_baseline,
    compute_normal_flow_patch,
    compute_normal_flow_patch_dso,
    compute_normal_flow_patch_dso_oriented,
    preprocess,
    select_strongest_gradient_points,
)
from dso_pixel_selector import (  # noqa: E402
    DSOSelectorConfig,
    build_dso_confidence,
    draw_selected_points,
    save_confidence_heatmap,
    select_dso_points,
)


REPO_ROOT = Path(__file__).resolve().parent
PAIR_RE = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)_flow\.npy$")
DISTANCE_BINS = [
    (0.0, 2.0, "0-2"),
    (2.0, 4.0, "2-4"),
    (4.0, 8.0, "4-8"),
    (8.0, 16.0, "8-16"),
    (16.0, 32.0, "16-32"),
    (32.0, float("inf"), ">32"),
]


@dataclass
class MethodStats:
    evaluated: int = 0
    valid: int = 0
    errors: list[float] = field(default_factory=list)
    epes: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def update(self, abs_error: float, epe: float, is_valid: bool) -> None:
        self.evaluated += 1
        if not is_valid:
            return
        self.valid += 1
        self.errors.append(float(abs_error))
        self.epes.append(float(epe))

    def as_dict(self) -> dict[str, float | int]:
        if not self.errors:
            return {
                "evaluated_points": self.evaluated,
                "valid_points": self.valid,
                "valid_fraction": 0.0,
                "mean_abs_normal_error": float("nan"),
                "median_abs_normal_error": float("nan"),
                "p90_abs_normal_error": float("nan"),
                "rmse": float("nan"),
                "aepe": float("nan"),
                "bad_0p5": float("nan"),
                "bad_1": float("nan"),
                "bad_2": float("nan"),
                "seconds": self.seconds,
            }
        errors = np.asarray(self.errors, dtype=np.float64)
        epes = np.asarray(self.epes, dtype=np.float64)
        return {
            "evaluated_points": self.evaluated,
            "valid_points": self.valid,
            "valid_fraction": self.valid / self.evaluated if self.evaluated else 0.0,
            "mean_abs_normal_error": float(np.mean(errors)),
            "median_abs_normal_error": float(np.median(errors)),
            "p90_abs_normal_error": float(np.percentile(errors, 90)),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "aepe": float(np.mean(epes)),
            "bad_0p5": float(np.mean(errors > 0.5)),
            "bad_1": float(np.mean(errors > 1.0)),
            "bad_2": float(np.mean(errors > 2.0)),
            "seconds": self.seconds,
        }


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
    parser.add_argument("--grid-step", type=int, default=80)
    parser.add_argument("--dev-pairs", type=int, default=40)
    parser.add_argument("--eval-pairs", type=int, default=80)
    parser.add_argument("--patch-radius", type=int, default=PATCH_RADIUS)
    parser.add_argument("--lambda-dso", type=float, default=DSO_WEIGHT_LAMBDA)
    parser.add_argument("--sigma-dso", type=float, default=DSO_SIGMA)
    parser.add_argument("--orientation-gamma", type=float, default=ORIENTATION_GAMMA)
    parser.add_argument("--selector-potential", type=int, default=3)
    parser.add_argument(
        "--spatial-sigma",
        type=float,
        default=0.0,
        help="0 disables optional spatial patch weighting.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip one-parameter development sweeps.",
    )
    return parser.parse_args()


def image_path(sequence_root: Path, frame_id: str) -> Path:
    return sequence_root / "image_left" / f"{frame_id}_left.png"


def flow_pairs(sequence_root: Path) -> list[tuple[str, str, Path]]:
    pairs = []
    for path in sorted((sequence_root / "flow").glob("*_flow.npy")):
        match = PAIR_RE.match(path.name)
        if match is None:
            continue
        pairs.append((match.group("first"), match.group("second"), path))
    if not pairs:
        raise FileNotFoundError(sequence_root / "flow")
    return pairs


def read_pair(sequence_root: Path, first: str, second: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_bgr = cv2.imread(str(image_path(sequence_root, first)), cv2.IMREAD_COLOR)
    second_bgr = cv2.imread(str(image_path(sequence_root, second)), cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise FileNotFoundError(image_path(sequence_root, first))
    if second_bgr is None:
        raise FileNotFoundError(image_path(sequence_root, second))
    return first_bgr, preprocess(first_bgr), preprocess(second_bgr)


def load_flow_gt(sequence_root: Path, first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    stem = f"{first}_{second}"
    flow = np.load(sequence_root / "flow" / f"{stem}_flow.npy").astype(np.float32)
    mask = np.load(sequence_root / "flow" / f"{stem}_mask.npy")
    return flow, mask > 0


def normal_gt_from_flow(
    flow: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_mag = np.sqrt(ix * ix + iy * iy)
    valid = grad_mag > GRAD_THRESHOLD
    nx = np.zeros_like(ix, dtype=np.float32)
    ny = np.zeros_like(iy, dtype=np.float32)
    nx[valid] = ix[valid] / grad_mag[valid]
    ny[valid] = iy[valid] / grad_mag[valid]
    gt_scalar = flow[..., 0] * nx + flow[..., 1] * ny
    gt_vector = np.stack((gt_scalar * nx, gt_scalar * ny), axis=-1)
    return gt_scalar.astype(np.float32), gt_vector.astype(np.float32), valid


def vector_and_scalar(vx: np.ndarray, vy: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.stack((vx, vy), axis=-1).astype(np.float32)
    scalar = (vx * nx + vy * ny).astype(np.float32)
    return vector, scalar


def method_defs(
    dso_confidence: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[str, callable]]:
    spatial_sigma = args.spatial_sigma if args.spatial_sigma > 0 else None
    return [
        (
            "classical_nf",
            lambda I0, I1, locations: compute_normal_flow_baseline(I0, I0, I1, 0.0, 0.0, 1.0),
        ),
        (
            "patch_nf",
            lambda I0, I1, locations: compute_normal_flow_patch(
                I0,
                I0,
                I1,
                0.0,
                0.0,
                1.0,
                locations=locations,
                patch_radius=args.patch_radius,
            ),
        ),
        (
            "dso_patch_lambda0",
            lambda I0, I1, locations: compute_normal_flow_patch_dso(
                I0,
                I0,
                I1,
                0.0,
                0.0,
                1.0,
                dso_confidence,
                locations=locations,
                patch_radius=args.patch_radius,
                lambda_dso=0.0,
                spatial_sigma=spatial_sigma,
            ),
        ),
        (
            "dso_patch_nf",
            lambda I0, I1, locations: compute_normal_flow_patch_dso(
                I0,
                I0,
                I1,
                0.0,
                0.0,
                1.0,
                dso_confidence,
                locations=locations,
                patch_radius=args.patch_radius,
                lambda_dso=args.lambda_dso,
                spatial_sigma=spatial_sigma,
            ),
        ),
        (
            "dso_oriented_patch_nf",
            lambda I0, I1, locations: compute_normal_flow_patch_dso_oriented(
                I0,
                I0,
                I1,
                0.0,
                0.0,
                1.0,
                dso_confidence,
                locations=locations,
                patch_radius=args.patch_radius,
                lambda_dso=args.lambda_dso,
                orientation_gamma=args.orientation_gamma,
                spatial_sigma=spatial_sigma,
            ),
        ),
    ]


def evaluate_subset(
    pairs: list[tuple[str, str, Path]],
    args: argparse.Namespace,
    split_name: str,
    write_rows: bool,
) -> tuple[dict, list[dict], dict, dict]:
    selector_config = DSOSelectorConfig(
        potential=args.selector_potential,
        dso_sigma=args.sigma_dso,
    )
    method_stats: dict[str, MethodStats] = defaultdict(MethodStats)
    distance_stats: dict[tuple[str, str], MethodStats] = defaultdict(MethodStats)
    region_stats: dict[tuple[str, str], MethodStats] = defaultdict(MethodStats)
    rows = []
    timing = {
        "selector_seconds": 0.0,
        "confidence_seconds": 0.0,
        "method_seconds": defaultdict(float),
    }
    lambda0_max_diff = 0.0

    debug_dir = args.output_root / "debug"
    plot_dir = args.output_root / "plots"
    debug_written = False

    for pair_index, (first, second, _flow_path) in enumerate(pairs):
        frame_bgr, I0, I1 = read_pair(args.sequence_root, first, second)
        ix, iy = compute_image_derivatives(I0)
        grad_mag = np.sqrt(ix * ix + iy * iy)
        flow, flow_valid = load_flow_gt(args.sequence_root, first, second)
        gt_scalar, gt_vector, gradient_valid = normal_gt_from_flow(flow, ix, iy)

        t0 = time.perf_counter()
        dso_mask, dso_points = select_dso_points(I0, selector_config)
        timing["selector_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        dso_confidence, dso_distance = build_dso_confidence(dso_mask, args.sigma_dso)
        timing["confidence_seconds"] += time.perf_counter() - t0

        if not debug_written:
            debug_dir.mkdir(parents=True, exist_ok=True)
            draw_selected_points(frame_bgr, dso_mask, debug_dir / "dso_selected_points.png")
            save_confidence_heatmap(dso_confidence, debug_dir / "dso_confidence_heatmap.png")
            debug_written = True

        eval_valid = flow_valid & gradient_valid
        locations = select_strongest_gradient_points(
            eval_valid,
            grad_mag,
            step=args.grid_step,
        )
        if not locations:
            continue

        nx = np.zeros_like(ix, dtype=np.float32)
        ny = np.zeros_like(iy, dtype=np.float32)
        nx[gradient_valid] = ix[gradient_valid] / grad_mag[gradient_valid]
        ny[gradient_valid] = iy[gradient_valid] / grad_mag[gradient_valid]

        method_outputs = {}
        for method_name, method_fn in method_defs(dso_confidence, args):
            t0 = time.perf_counter()
            result = method_fn(I0, I1, locations)
            elapsed = time.perf_counter() - t0
            timing["method_seconds"][method_name] += elapsed
            method_stats[method_name].seconds += elapsed
            if method_name == "classical_nf":
                vx, vy, _mag, valid, *_rest = result
                photo = np.full_like(I0, np.nan, dtype=np.float32)
            else:
                vx, vy, _mag, valid, _confidence, photo, *_rest = result
            pred_vector, pred_scalar = vector_and_scalar(vx, vy, nx, ny)
            method_outputs[method_name] = (pred_scalar, pred_vector, valid, photo)

        patch_scalar = method_outputs["patch_nf"][0]
        lambda0_scalar = method_outputs["dso_patch_lambda0"][0]
        patch_valid = method_outputs["patch_nf"][2]
        lambda0_valid = method_outputs["dso_patch_lambda0"][2]
        compare_mask = patch_valid & lambda0_valid
        if np.any(compare_mask):
            lambda0_max_diff = max(
                lambda0_max_diff,
                float(np.max(np.abs(patch_scalar[compare_mask] - lambda0_scalar[compare_mask]))),
            )

        for y, x in locations:
            is_dso_point = bool(dso_mask[y, x] > 0)
            distance = float(dso_distance[y, x])
            confidence = float(dso_confidence[y, x])
            gt_s = float(gt_scalar[y, x])
            gt_v = gt_vector[y, x]

            if distance <= 1e-6:
                region = "exact_dso"
            elif distance <= args.sigma_dso:
                region = "near_non_dso"
            else:
                region = "distant"

            for method_name, (pred_scalar, pred_vector, valid, photo) in method_outputs.items():
                is_valid = bool(valid[y, x])
                estimated = float(pred_scalar[y, x]) if is_valid else float("nan")
                abs_error = abs(estimated - gt_s) if is_valid else float("nan")
                epe = (
                    float(np.linalg.norm(pred_vector[y, x] - gt_v))
                    if is_valid
                    else float("nan")
                )

                method_stats[method_name].update(abs_error, epe, is_valid)
                bin_label = distance_bin(distance)
                distance_stats[(method_name, bin_label)].update(abs_error, epe, is_valid)
                region_stats[(method_name, region)].update(abs_error, epe, is_valid)

                if write_rows:
                    rows.append(
                        {
                            "sequence": "Office",
                            "split": split_name,
                            "frame": first,
                            "x": x,
                            "y": y,
                            "method": method_name,
                            "is_dso_point": int(is_dso_point),
                            "distance_to_nearest_dso": distance,
                            "dso_confidence": confidence,
                            "gradient_magnitude": float(grad_mag[y, x]),
                            "gt_normal_displacement": gt_s,
                            "estimated_normal_displacement": estimated,
                            "absolute_normal_error": abs_error,
                            "normal_epe": epe,
                            "photometric_error": float(photo[y, x]) if np.isfinite(photo[y, x]) else "",
                            "valid": int(is_valid),
                            "runtime": timing["method_seconds"][method_name],
                            "lambda": args.lambda_dso if method_name != "dso_patch_lambda0" else 0.0,
                            "sigma": args.sigma_dso,
                            "patch_radius": args.patch_radius,
                            "orientation_gamma": (
                                args.orientation_gamma
                                if method_name == "dso_oriented_patch_nf"
                                else 0.0
                            ),
                        }
                    )

        if pair_index % 20 == 0 or pair_index + 1 == len(pairs):
            print(f"{split_name}: evaluated {pair_index + 1}/{len(pairs)} pairs", flush=True)

    summary = {
        "methods": {name: stats.as_dict() for name, stats in sorted(method_stats.items())},
        "distance_bins": {
            f"{method}:{bin_label}": stats.as_dict()
            for (method, bin_label), stats in sorted(distance_stats.items())
        },
        "regions": {
            f"{method}:{region}": stats.as_dict()
            for (method, region), stats in sorted(region_stats.items())
        },
        "timing": {
            "selector_ms_per_frame": 1000.0 * timing["selector_seconds"] / max(1, len(pairs)),
            "confidence_ms_per_frame": 1000.0 * timing["confidence_seconds"] / max(1, len(pairs)),
            "method_ms_per_frame": {
                name: 1000.0 * seconds / max(1, len(pairs))
                for name, seconds in timing["method_seconds"].items()
            },
        },
        "lambda0_patch_max_abs_diff": lambda0_max_diff,
    }
    return summary, rows, method_stats, distance_stats


def distance_bin(distance: float) -> str:
    for lo, hi, label in DISTANCE_BINS:
        if lo <= distance < hi:
            return label
    return ">32"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict) -> None:
    rows = []
    for method, stats in summary["methods"].items():
        rows.append({"section": "method", "name": method, **stats})
    for key, stats in summary["distance_bins"].items():
        rows.append({"section": "distance_bin", "name": key, **stats})
    for key, stats in summary["regions"].items():
        rows.append({"section": "region", "name": key, **stats})
    write_csv(path, rows)


def plot_summary(output_root: Path, summary: dict) -> None:
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["methods"].keys())
    mean_errors = [summary["methods"][m]["mean_abs_normal_error"] for m in methods]
    median_errors = [summary["methods"][m]["median_abs_normal_error"] for m in methods]

    for values, name, ylabel in [
        (mean_errors, "mean_normal_error_by_method.png", "Mean abs normal error"),
        (median_errors, "median_normal_error_by_method.png", "Median abs normal error"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(methods, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=160)
        plt.close(fig)

    labels = [label for *_rest, label in DISTANCE_BINS]
    fig, ax = plt.subplots(figsize=(8, 4))
    for method in ["patch_nf", "dso_patch_nf", "dso_oriented_patch_nf"]:
        vals = []
        for label in labels:
            key = f"{method}:{label}"
            vals.append(summary["distance_bins"].get(key, {}).get("mean_abs_normal_error", np.nan))
        ax.plot(labels, vals, marker="o", label=method)
    ax.set_ylabel("Mean abs normal error")
    ax.set_xlabel("Distance to nearest DSO point (px)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "error_vs_distance_to_dso.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = []
    for label in labels:
        patch_key = f"patch_nf:{label}"
        dso_key = f"dso_patch_nf:{label}"
        patch_err = summary["distance_bins"].get(patch_key, {}).get("mean_abs_normal_error", np.nan)
        dso_err = summary["distance_bins"].get(dso_key, {}).get("mean_abs_normal_error", np.nan)
        vals.append(patch_err - dso_err)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.bar(labels, vals)
    ax.set_ylabel("Patch error - DSO patch error")
    ax.set_xlabel("Distance to nearest DSO point (px)")
    fig.tight_layout()
    fig.savefig(plot_dir / "dso_improvement_vs_distance.png", dpi=160)
    plt.close(fig)


def run_one_parameter_sweeps(
    dev_pairs: list[tuple[str, str, Path]],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    sweep_specs = [
        ("lambda", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]),
        ("sigma", [2.0, 4.0, 8.0, 16.0]),
        ("patch_radius", [1, 2, 3]),
        ("orientation_gamma", [0.0, 1.0, 2.0, 4.0]),
    ]
    for parameter, values in sweep_specs:
        for value in values:
            sweep_args = argparse.Namespace(**vars(args))
            if parameter == "lambda":
                sweep_args.lambda_dso = value
            elif parameter == "sigma":
                sweep_args.sigma_dso = value
            elif parameter == "patch_radius":
                sweep_args.patch_radius = value
            elif parameter == "orientation_gamma":
                sweep_args.orientation_gamma = value

            summary, _point_rows, _stats, _dist = evaluate_subset(
                dev_pairs,
                sweep_args,
                split_name=f"dev_{parameter}_{value}",
                write_rows=False,
            )
            for method in ["patch_nf", "dso_patch_nf", "dso_oriented_patch_nf"]:
                stats = summary["methods"].get(method, {})
                rows.append(
                    {
                        "sweep_parameter": parameter,
                        "sweep_value": value,
                        "method": method,
                        "mean_abs_normal_error": stats.get("mean_abs_normal_error", np.nan),
                        "median_abs_normal_error": stats.get("median_abs_normal_error", np.nan),
                        "rmse": stats.get("rmse", np.nan),
                        "aepe": stats.get("aepe", np.nan),
                        "valid_fraction": stats.get("valid_fraction", np.nan),
                    }
                )
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    eval_summary: dict,
    sweep_rows: list[dict],
) -> None:
    methods = eval_summary["methods"]
    rows = []
    for method in [
        "classical_nf",
        "patch_nf",
        "dso_patch_lambda0",
        "dso_patch_nf",
        "dso_oriented_patch_nf",
    ]:
        stats = methods[method]
        rows.append(
            f"| {method} | {stats['valid_points']:,} | "
            f"{stats['mean_abs_normal_error']:.4f} | "
            f"{stats['median_abs_normal_error']:.4f} | "
            f"{stats['p90_abs_normal_error']:.4f} | "
            f"{stats['rmse']:.4f} | {stats['aepe']:.4f} | "
            f"{stats['bad_1']:.3f} |"
        )

    rq2 = (
        methods["patch_nf"]["mean_abs_normal_error"]
        - methods["dso_patch_nf"]["mean_abs_normal_error"]
    )
    rq5 = (
        methods["dso_patch_nf"]["mean_abs_normal_error"]
        - methods["dso_oriented_patch_nf"]["mean_abs_normal_error"]
    )
    near_patch = eval_summary["regions"].get("patch_nf:near_non_dso", {})
    near_dso = eval_summary["regions"].get("dso_patch_nf:near_non_dso", {})
    near_delta = near_patch.get("mean_abs_normal_error", np.nan) - near_dso.get(
        "mean_abs_normal_error",
        np.nan,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# DSO-Weighted Direct Patch Normal Flow Report",
                "",
                "## Protocol",
                "",
                f"- Sequence: TartanAir Office",
                f"- Development pairs: {args.dev_pairs}",
                f"- Evaluation pairs: {args.eval_pairs}",
                f"- Grid step: {args.grid_step}",
                f"- Patch radius: {args.patch_radius}",
                f"- DSO sigma: {args.sigma_dso}",
                f"- DSO lambda: {args.lambda_dso}",
                f"- Orientation gamma: {args.orientation_gamma}",
                "",
                "The DSO selector here is a Python DSO-style frontend selector: adaptive local gradient thresholds, deterministic block selection, and directional diversity. It does not use DSO pose, depth, tracking, or bundle adjustment.",
                "",
                "## Ablation",
                "",
                "| Method | Patch | DSO selector | DSO spatial weighting | Orientation gating |",
                "|---|---:|---:|---:|---:|",
                "| Classical NF | No | No | No | No |",
                "| Patch NF | Yes | No | No | No |",
                "| DSO Patch lambda=0 | Yes | Yes | No | No |",
                "| DSO Patch | Yes | Yes | Yes | No |",
                "| DSO Oriented Patch | Yes | Yes | Yes | Yes |",
                "",
                "## Evaluation Results",
                "",
                "| Method | Valid points | Mean error | Median error | P90 error | RMSE | AEPE | Bad >1 px |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *rows,
                "",
                "## Runtime",
                "",
                f"- DSO selector: {eval_summary['timing']['selector_ms_per_frame']:.3f} ms/frame",
                f"- Confidence map: {eval_summary['timing']['confidence_ms_per_frame']:.3f} ms/frame",
                *[
                    f"- {name}: {ms:.3f} ms/frame"
                    for name, ms in eval_summary["timing"]["method_ms_per_frame"].items()
                ],
                "",
                "## Sanity Checks",
                "",
                f"- `dso_patch_lambda0` max scalar difference from ordinary `patch_nf`: {eval_summary['lambda0_patch_max_abs_diff']:.12g}",
                "",
                "## Research Questions",
                "",
                f"- RQ1: Patch NF vs classical NF: patch mean error is {methods['patch_nf']['mean_abs_normal_error']:.4f}, classical is {methods['classical_nf']['mean_abs_normal_error']:.4f}.",
                f"- RQ2: DSO weighting vs patch NF: mean-error delta `patch - dso_patch` is {rq2:.4f}. Positive means DSO weighting helped.",
                f"- RQ3: Nearby non-DSO pixels: delta `patch - dso_patch` is {near_delta:.4f}.",
                "- RQ4: See `plots/error_vs_distance_to_dso.png` and `plots/dso_improvement_vs_distance.png`.",
                f"- RQ5: Orientation gating delta `dso_patch - dso_oriented` is {rq5:.4f}. Positive means orientation gating helped.",
                "- RQ6: DSO weighting adds selector/confidence overhead plus the weighted patch estimator cost shown above.",
                "",
                "## Parameter Sweep",
                "",
                f"Saved {len(sweep_rows)} one-parameter development-sweep rows to `summary_sweep.csv`.",
                "",
                "## Outputs",
                "",
                "- `raw_results.csv`: one row per evaluated point and method.",
                "- `summary.csv`: aggregate, distance-bin, and DSO-region metrics.",
                "- `summary.json`: full machine-readable result.",
                "- `debug/dso_selected_points.png`: selector overlay on the exact NF image coordinates.",
                "- `debug/dso_confidence_heatmap.png`: DSO confidence field.",
                "- `plots/`: summary plots.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.sequence_root = args.sequence_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pairs = flow_pairs(args.sequence_root)
    dev_pairs = pairs[: args.dev_pairs]
    eval_pairs = pairs[args.dev_pairs : args.dev_pairs + args.eval_pairs]
    if not eval_pairs:
        raise ValueError("No evaluation pairs selected")

    sweep_rows = []
    if not args.skip_sweep and dev_pairs:
        sweep_rows = run_one_parameter_sweeps(dev_pairs, args)
        write_csv(args.output_root / "summary_sweep.csv", sweep_rows)

    eval_summary, point_rows, _method_stats, _distance_stats = evaluate_subset(
        eval_pairs,
        args,
        split_name="eval",
        write_rows=True,
    )

    result = {
        "sequence_root": str(args.sequence_root),
        "dev_pairs": len(dev_pairs),
        "eval_pairs": len(eval_pairs),
        "parameters": {
            "grid_step": args.grid_step,
            "patch_radius": args.patch_radius,
            "lambda_dso": args.lambda_dso,
            "sigma_dso": args.sigma_dso,
            "orientation_gamma": args.orientation_gamma,
            "selector_potential": args.selector_potential,
            "spatial_sigma": args.spatial_sigma,
        },
        "eval": eval_summary,
        "sweep_rows": sweep_rows,
    }

    (args.output_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_root / "raw_results.csv", point_rows)
    write_summary_csv(args.output_root / "summary.csv", eval_summary)
    plot_summary(args.output_root, eval_summary)
    write_report(args.output_root / "DSO_WEIGHTED_PATCH_NF_REPORT.md", args, eval_summary, sweep_rows)

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
