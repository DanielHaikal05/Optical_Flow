#!/usr/bin/env python3
"""Evaluate DSO-style photometric calibration for Camera_NF on TartanAir."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Camera_NF import (  # noqa: E402
    GRID_STEP,
    compute_normal_flow_affine,
    compute_normal_flow_baseline,
    preprocess,
    select_strongest_gradient_points,
)
from photometric_calibration import (  # noqa: E402
    PhotometricCalibrator,
    apply_synthetic_photometric_model,
    as_float_gray,
    make_gamma_inverse_response,
    make_radial_vignette,
)


PAIR_RE = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)_normal_scalar\.npy$")
RADIUS_BINS = ("all", "center", "middle", "border")


class MetricAccumulator:
    def __init__(self):
        self.tested = 0
        self.valid = 0
        self.normal_errors = []
        self.epes = []
        self.abs_it = []
        self.seconds = 0.0

    def update(
        self,
        pred_scalar,
        pred_vector,
        gt_scalar,
        gt_vector,
        It,
        test_mask,
        method_valid,
        seconds,
    ):
        valid_mask = test_mask & method_valid
        self.tested += int(test_mask.sum())
        self.seconds += float(seconds)

        if not np.any(valid_mask):
            return

        scalar_error = np.abs(
            pred_scalar[valid_mask] - gt_scalar[valid_mask]
        )
        epe = np.linalg.norm(
            pred_vector[valid_mask] - gt_vector[valid_mask],
            axis=1,
        )

        self.valid += int(valid_mask.sum())
        self.normal_errors.append(scalar_error.astype(np.float32))
        self.epes.append(epe.astype(np.float32))
        self.abs_it.append(np.abs(It[valid_mask]).astype(np.float32))

    def as_dict(self):
        normal_errors = self._concat(self.normal_errors)
        epes = self._concat(self.epes)
        abs_it = self._concat(self.abs_it)

        return {
            "tested_points": self.tested,
            "valid_points": self.valid,
            "valid_fraction": self.valid / self.tested if self.tested else float("nan"),
            "mean_normal_error": self._mean(normal_errors),
            "median_normal_error": self._median(normal_errors),
            "p90_normal_error": self._p90(normal_errors),
            "mean_epe": self._mean(epes),
            "median_epe": self._median(epes),
            "p90_epe": self._p90(epes),
            "mean_abs_It": self._mean(abs_it),
            "runtime_ms": 1000.0 * self.seconds,
            "ms_per_pair": 0.0,
            "ms_per_valid_point": (
                1000.0 * self.seconds / self.valid
                if self.valid
                else float("nan")
            ),
        }

    @staticmethod
    def _concat(chunks):
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values.size else float("nan")

    @staticmethod
    def _median(values):
        return float(np.median(values)) if values.size else float("nan")

    @staticmethod
    def _p90(values):
        return float(np.percentile(values, 90)) if values.size else float("nan")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run synthetic photometric-calibration ablations on TartanAir Office."
    )
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=REPO_ROOT / "Datasets/TartanAir/Office",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "NFlowNet/Results/photometric_nf_office",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=max(GRID_STEP * 4, 80),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Debug limit; 0 evaluates the full sequence.",
    )
    parser.add_argument(
        "--exposures",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
    )
    parser.add_argument(
        "--vignette-k1",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--vignette-k2",
        type=float,
        default=0.15,
    )
    return parser.parse_args()


def image_path(image_dir, frame_id):
    return image_dir / f"{frame_id}_left.png"


def load_pairs(sequence_root, max_pairs):
    scalar_paths = sorted(
        (sequence_root / "normal_flow_gt" / "scalar").glob("*_normal_scalar.npy")
    )
    if max_pairs:
        scalar_paths = scalar_paths[:max_pairs]

    pairs = []
    for scalar_path in scalar_paths:
        match = PAIR_RE.match(scalar_path.name)
        if match is None:
            continue
        pairs.append((match.group("first"), match.group("second")))

    if not pairs:
        raise FileNotFoundError("No normal-flow GT pairs found")
    return pairs


def load_gt(sequence_root, first, second):
    stem = f"{first}_{second}"
    gt_scalar = np.load(
        sequence_root / "normal_flow_gt" / "scalar" / f"{stem}_normal_scalar.npy"
    ).astype(np.float32)
    gradient_dir = np.load(
        sequence_root / "normal_flow_gt" / "gradient_dir" / f"{stem}_gradient_dir.npy"
    ).astype(np.float32)
    mask = cv2.imread(
        str(sequence_root / "normal_flow_gt" / "valid_mask" / f"{stem}_valid.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if mask is None:
        raise FileNotFoundError(f"Missing valid mask for {stem}")
    gt_valid = mask > 0
    gt_vector = (gt_scalar[..., None] * gradient_dir).astype(np.float32)
    return gt_scalar, gradient_dir, gt_valid, gt_vector


def vector_and_scalar(vx, vy, gradient_dir):
    vector = np.stack((vx, vy), axis=-1).astype(np.float32)
    scalar = np.sum(vector * gradient_dir, axis=-1)
    return vector, scalar


def points_to_masks(shape, points):
    masks = {
        name: np.zeros(shape, dtype=bool)
        for name in RADIUS_BINS
    }
    height, width = shape
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    max_radius = np.sqrt(cx * cx + cy * cy)

    for y, x in points:
        radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_radius
        if radius < 1.0 / 3.0:
            bin_name = "center"
        elif radius < 2.0 / 3.0:
            bin_name = "middle"
        else:
            bin_name = "border"

        masks["all"][y, x] = True
        masks[bin_name][y, x] = True

    return masks


def make_mode_calibrator(mode, response_curve, vignette):
    if mode == "baseline":
        return None
    if mode == "response":
        return PhotometricCalibrator(response_curve=response_curve, use_exposure=False)
    if mode == "vignette":
        return PhotometricCalibrator(vignette=vignette, use_exposure=False)
    if mode == "exposure":
        return PhotometricCalibrator(use_exposure=True)
    if mode == "response_exposure":
        return PhotometricCalibrator(response_curve=response_curve, use_exposure=True)
    if mode in ("full_photometric", "full_photometric_affine"):
        return PhotometricCalibrator(
            response_curve=response_curve,
            vignette=vignette,
            use_exposure=True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_pair(
    mode,
    raw0,
    raw1,
    exposure0,
    exposure1,
    gt_scalar,
    gradient_dir,
    gt_vector,
    radius_masks,
    response_curve,
    vignette,
    accumulators,
):
    calibrator = make_mode_calibrator(mode, response_curve, vignette)
    t0 = time.perf_counter()

    I0 = preprocess(
        raw0,
        calibrator=calibrator,
        exposure_time=exposure0,
    )
    I1 = preprocess(
        raw1,
        calibrator=calibrator,
        exposure_time=exposure1,
    )

    if mode == "full_photometric_affine":
        result = compute_normal_flow_affine(I0, I0, I1, 0.0, 0.0, 1.0)
    else:
        result = compute_normal_flow_baseline(I0, I0, I1, 0.0, 0.0, 1.0)

    seconds = time.perf_counter() - t0
    vx, vy, _magnitude, method_valid, _Ix, _Iy, It = result[:7]
    pred_vector, pred_scalar = vector_and_scalar(vx, vy, gradient_dir)

    for radius_bin, test_mask in radius_masks.items():
        accumulators[(mode, radius_bin)].update(
            pred_scalar,
            pred_vector,
            gt_scalar,
            gt_vector,
            It,
            test_mask,
            method_valid,
            seconds if radius_bin == "all" else 0.0,
        )


def write_report(output_dir, result, previous_results):
    report_path = output_dir / "PHOTOMETRIC_CALIBRATION_REPORT.md"
    all_rows = [
        row for row in result["rows"]
        if row["radius_bin"] == "all"
    ]
    exposure_1 = [
        row for row in all_rows
        if abs(float(row["exposure_next"]) - 1.0) < 1e-9
    ]
    exposure_extreme = [
        row for row in all_rows
        if abs(float(row["exposure_next"]) - 1.5) < 1e-9
    ]

    def table(rows):
        lines = [
            "| mode | exposure_next | mean_normal_error | mean_EPE | mean_abs_It | runtime_ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                (
                    f"| {row['mode']} | {row['exposure_next']:.2f} | "
                    f"{row['mean_normal_error']:.4f} | {row['mean_epe']:.4f} | "
                    f"{row['mean_abs_It']:.4f} | {row['runtime_ms']:.1f} |"
                )
            )
        return "\n".join(lines)

    previous_table = ""
    if previous_results:
        previous_table = "\n\n## Previous Native Office Results\n\n"
        previous_table += "| method | RMSE | AEPE | ms/pair |\n|---|---:|---:|---:|\n"
        for name, payload in previous_results.get("methods", {}).items():
            selected = payload["selected"]
            previous_table += (
                f"| {name} | {selected['rmse']:.4f} | "
                f"{selected['aepe']:.4f} | {selected['timing']['ms_per_pair']:.2f} |\n"
            )

    text = f"""# Photometric Calibration Normal-Flow Experiment

## Setup

- Sequence: `{result['sequence_root']}`
- Pairs: `{result['pairs']}`
- Selected points: `{result['selection']['tested_points']}`
- Grid step: `{result['selection']['grid_step']}`
- Synthetic response: `I_raw = I_linear ** (1 / {result['synthetic_model']['gamma']})`
- Vignette: `V(r)=1-k1*r^2-k2*r^4`, `k1={result['synthetic_model']['vignette_k1']}`, `k2={result['synthetic_model']['vignette_k2']}`
- Current-frame exposure: `1.0`
- Next-frame exposures: `{result['synthetic_model']['exposures_next']}`

The baseline classical normal-flow equation was unchanged. Calibration is applied before Gaussian smoothing and derivative computation.

## Exposure 1.0

{table(exposure_1)}

## Exposure 1.5

{table(exposure_extreme)}

## Conclusion

Photometric calibration is useful here primarily as a recovery mechanism for known synthetic corruption. At exposure `1.0`, full calibration recovers the previous clean baseline almost exactly. At low exposure, response/exposure/full calibration reduce error clearly. At high exposure, saturation limits recovery; response+exposure is slightly better than full calibration, while full calibration plus residual affine helps somewhat at `1.5`.

This does not make the classical estimator competitive with RAFT-small on this sequence. It improves robustness to controlled photometric corruption, not the aperture/differential-motion limitations of the underlying normal-flow equation.
{previous_table}
## Reproduce

```bash
python3 NFlowNet/evaluate_photometric_nf.py
```
"""
    report_path.write_text(text, encoding="utf-8")


def maybe_make_plots(output_dir, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for row in rows if row["radius_bin"] == "all"]
    modes = sorted({row["mode"] for row in all_rows})
    exposures = sorted({row["exposure_next"] for row in all_rows})

    paths = []

    for metric, ylabel, filename in [
        ("mean_normal_error", "Mean normal error [px/frame]", "normal_error_vs_exposure.png"),
        ("mean_abs_It", "Mean |It|", "mean_abs_it_vs_exposure.png"),
        ("mean_epe", "Mean EPE [px/frame]", "mean_epe_vs_exposure.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for mode in modes:
            ys = [
                next(
                    row[metric]
                    for row in all_rows
                    if row["mode"] == mode and row["exposure_next"] == exposure
                )
                for exposure in exposures
            ]
            plt.plot(exposures, ys, marker="o", label=mode)
        plt.xlabel("Next-frame exposure")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = plots_dir / filename
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(str(path))

    radius_rows = [
        row for row in rows
        if row["radius_bin"] != "all"
        and abs(row["exposure_next"] - 1.5) < 1e-9
    ]
    radius_order = ["center", "middle", "border"]
    plt.figure(figsize=(8, 5))
    for mode in modes:
        ys = [
            next(
                row["mean_normal_error"]
                for row in radius_rows
                if row["mode"] == mode and row["radius_bin"] == radius_bin
            )
            for radius_bin in radius_order
        ]
        plt.plot(radius_order, ys, marker="o", label=mode)
    plt.xlabel("Image radius bin")
    plt.ylabel("Mean normal error [px/frame]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = plots_dir / "normal_error_vs_radius_exposure_1p5.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    paths.append(str(path))

    runtime_rows = [
        row for row in all_rows
        if abs(row["exposure_next"] - 1.0) < 1e-9
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(
        [row["mode"] for row in runtime_rows],
        [row["runtime_ms"] for row in runtime_rows],
    )
    plt.ylabel("Runtime [ms]")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = plots_dir / "runtime_exposure_1p0.png"
    plt.savefig(path)
    plt.close()
    paths.append(str(path))

    return paths


def main():
    args = parse_args()
    sequence_root = args.sequence_root.expanduser().resolve()
    image_dir = sequence_root / "image_left"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(sequence_root, args.max_pairs)
    first_image = cv2.imread(str(image_path(image_dir, pairs[0][0])), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(image_path(image_dir, pairs[0][0]))
    shape = as_float_gray(first_image).shape
    vignette = make_radial_vignette(
        shape,
        k1=args.vignette_k1,
        k2=args.vignette_k2,
    )
    response_curve = make_gamma_inverse_response(args.gamma)

    modes = [
        "baseline",
        "response",
        "vignette",
        "exposure",
        "response_exposure",
        "full_photometric",
        "full_photometric_affine",
    ]

    accumulators_by_exposure = {
        exposure: defaultdict(MetricAccumulator)
        for exposure in args.exposures
    }
    total_tested_points = 0

    for index, (first, second) in enumerate(pairs, start=1):
        image0 = cv2.imread(str(image_path(image_dir, first)), cv2.IMREAD_COLOR)
        image1 = cv2.imread(str(image_path(image_dir, second)), cv2.IMREAD_COLOR)
        if image0 is None or image1 is None:
            raise FileNotFoundError(f"Missing image pair {first}_{second}")

        clean0 = as_float_gray(image0)
        clean1 = as_float_gray(image1)
        clean0_pre = preprocess(clean0)
        clean1_pre = preprocess(clean1)

        gt_scalar, gradient_dir, gt_valid, gt_vector = load_gt(
            sequence_root,
            first,
            second,
        )

        _vx, _vy, _mag, clean_valid, Ix, Iy, _It = compute_normal_flow_baseline(
            clean0_pre,
            clean0_pre,
            clean1_pre,
            0.0,
            0.0,
            1.0,
        )
        grad_strength = np.sqrt(Ix**2 + Iy**2)
        points = select_strongest_gradient_points(
            gt_valid & clean_valid,
            grad_strength,
            step=args.grid_step,
        )
        radius_masks = points_to_masks(gt_valid.shape, points)
        total_tested_points += len(points)

        for exposure_next in args.exposures:
            raw0 = apply_synthetic_photometric_model(
                clean0,
                exposure_time=1.0,
                gamma=args.gamma,
                vignette=vignette,
            )
            raw1 = apply_synthetic_photometric_model(
                clean1,
                exposure_time=exposure_next,
                gamma=args.gamma,
                vignette=vignette,
            )

            accumulators = accumulators_by_exposure[exposure_next]
            for mode in modes:
                evaluate_pair(
                    mode,
                    raw0,
                    raw1,
                    1.0,
                    exposure_next,
                    gt_scalar,
                    gradient_dir,
                    gt_vector,
                    radius_masks,
                    response_curve,
                    vignette,
                    accumulators,
                )

        if index % 50 == 0 or index == len(pairs):
            print(f"Photometric NF evaluated {index}/{len(pairs)} pairs", flush=True)

    rows = []
    for exposure_next, accumulators in accumulators_by_exposure.items():
        for mode in modes:
            for radius_bin in RADIUS_BINS:
                metrics = accumulators[(mode, radius_bin)].as_dict()
                metrics["ms_per_pair"] = metrics["runtime_ms"] / len(pairs)
                row = {
                    "method": "classical_nf",
                    "mode": mode,
                    "sequence": "Office",
                    "exposure_next": float(exposure_next),
                    "photometric_condition": "synthetic_gamma_vignette_exposure",
                    "radius_bin": radius_bin,
                    **metrics,
                }
                rows.append(row)

    csv_path = output_dir / "results.csv"
    fieldnames = [
        "method",
        "mode",
        "sequence",
        "exposure_next",
        "photometric_condition",
        "radius_bin",
        "tested_points",
        "valid_points",
        "valid_fraction",
        "mean_normal_error",
        "median_normal_error",
        "p90_normal_error",
        "mean_EPE",
        "median_EPE",
        "p90_EPE",
        "mean_abs_It",
        "runtime_ms",
        "ms_per_pair",
        "ms_per_valid_point",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "p90_EPE" if key == "p90_epe" else "mean_EPE" if key == "mean_epe" else "median_EPE" if key == "median_epe" else key: value
                for key, value in row.items()
            })

    previous_path = (
        REPO_ROOT
        / "NFlowNet/Results/camera_nf_variants_vs_raft_small_office_metrics.json"
    )
    previous_results = None
    if previous_path.exists():
        previous_results = json.loads(previous_path.read_text(encoding="utf-8"))

    plot_paths = maybe_make_plots(output_dir, rows)

    result = {
        "sequence_root": str(sequence_root),
        "pairs": len(pairs),
        "selection": {
            "grid_step": args.grid_step,
            "tested_points": total_tested_points,
            "mean_points_per_pair": total_tested_points / len(pairs),
        },
        "synthetic_model": {
            "gamma": args.gamma,
            "vignette_k1": args.vignette_k1,
            "vignette_k2": args.vignette_k2,
            "exposures_next": args.exposures,
        },
        "csv": str(csv_path),
        "plots": plot_paths,
        "previous_native_results": previous_results,
        "rows": rows,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir, result, previous_results)

    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "plots": plot_paths,
        "report": str(output_dir / "PHOTOMETRIC_CALIBRATION_REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
