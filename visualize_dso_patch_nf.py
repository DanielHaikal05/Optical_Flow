#!/usr/bin/env python3
"""Generate diagnostics from DSO-weighted patch NF CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/dso_weighted_patch_nf",
    )
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=REPO_ROOT / "Datasets/TartanAir/Office",
    )
    parser.add_argument("--frame", default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return np.nan
    return float(value)


def plot_binned(
    rows: list[dict],
    x_key: str,
    output_path: Path,
    xlabel: str,
    bins: np.ndarray,
) -> None:
    methods = ["patch_nf", "dso_patch_nf", "dso_oriented_patch_nf"]
    fig, ax = plt.subplots(figsize=(8, 4))
    centers = 0.5 * (bins[:-1] + bins[1:])

    for method in methods:
        xs = np.asarray(
            [as_float(r, x_key) for r in rows if r["method"] == method and r["valid"] == "1"],
            dtype=np.float64,
        )
        errs = np.asarray(
            [
                as_float(r, "absolute_normal_error")
                for r in rows
                if r["method"] == method and r["valid"] == "1"
            ],
            dtype=np.float64,
        )
        values = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (xs >= lo) & (xs < hi)
            values.append(float(np.mean(errs[mask])) if np.any(mask) else np.nan)
        ax.plot(centers, values, marker="o", label=method)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean abs normal error")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sweep(sweep_rows: list[dict], parameter: str, output_path: Path) -> None:
    methods = ["patch_nf", "dso_patch_nf", "dso_oriented_patch_nf"]
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False

    for method in methods:
        pairs = [
            (float(r["sweep_value"]), float(r["mean_abs_normal_error"]))
            for r in sweep_rows
            if r["sweep_parameter"] == parameter and r["method"] == method
        ]
        pairs.sort(key=lambda item: item[0])
        if not pairs:
            continue
        plotted = True
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", label=method)

    ax.set_xlabel(parameter)
    ax.set_ylabel("Mean abs normal error")
    if plotted:
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_runtime(summary: dict, output_path: Path) -> None:
    timing = summary["eval"]["timing"]
    method_ms = timing["method_ms_per_frame"]
    labels = ["selector", "confidence", *method_ms.keys()]
    values = [
        timing["selector_ms_per_frame"],
        timing["confidence_ms_per_frame"],
        *method_ms.values(),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values)
    ax.set_ylabel("ms/frame")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_arrow_overlay(
    image_bgr: np.ndarray,
    rows: list[dict],
    method: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    output = image_bgr.copy()
    method_rows = [
        r
        for r in rows
        if r["method"] == method and r["valid"] == "1"
    ]
    for row in method_rows:
        x = int(row["x"])
        y = int(row["y"])
        d_est = as_float(row, "estimated_normal_displacement")
        d_gt = as_float(row, "gt_normal_displacement")
        if not np.isfinite(d_est):
            continue
        # Use scalar-only arrows horizontally for compact diagnostics. The
        # quantitative CSV carries the true scalar values.
        length = int(np.clip(d_est, -20, 20))
        cv2.arrowedLine(output, (x, y), (x + length, y), color, 1, tipLength=0.25)
        cv2.circle(output, (x, y), 2, (255, 255, 255), -1)
        if np.isfinite(d_gt):
            cv2.circle(output, (x, y), 1, (0, 0, 0), -1)
    return output


def make_debug_panel(args: argparse.Namespace, rows: list[dict]) -> None:
    frame = args.frame
    if not frame:
        frame = rows[0]["frame"]
    frame_rows = [r for r in rows if r["frame"] == frame]
    if not frame_rows:
        return

    image_path = args.sequence_root / "image_left" / f"{frame}_left.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return

    selected_path = args.results_root / "debug/dso_selected_points.png"
    confidence_path = args.results_root / "debug/dso_confidence_heatmap.png"
    selected = cv2.imread(str(selected_path), cv2.IMREAD_COLOR)
    confidence = cv2.imread(str(confidence_path), cv2.IMREAD_COLOR)
    if selected is None:
        selected = image.copy()
    if confidence is None:
        confidence = image.copy()

    patch = draw_arrow_overlay(image, frame_rows, "patch_nf", (0, 255, 255))
    dso = draw_arrow_overlay(image, frame_rows, "dso_patch_nf", (0, 255, 0))
    oriented = draw_arrow_overlay(image, frame_rows, "dso_oriented_patch_nf", (255, 0, 0))

    error_map = image.copy()
    for row in frame_rows:
        if row["method"] != "dso_patch_nf" or row["valid"] != "1":
            continue
        err = as_float(row, "absolute_normal_error")
        color = cv2.applyColorMap(
            np.uint8([[np.clip(err / 30.0 * 255.0, 0, 255)]]),
            cv2.COLORMAP_INFERNO,
        )[0, 0]
        cv2.circle(error_map, (int(row["x"]), int(row["y"])), 3, tuple(int(c) for c in color), -1)

    top = np.hstack([image, selected, confidence])
    bottom = np.hstack([patch, dso, oriented])
    panel = np.vstack([top, bottom])
    out = args.results_root / "debug/dso_patch_nf_debug_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel)
    cv2.imwrite(str(args.results_root / "debug/dso_patch_nf_error_points.png"), error_map)


def main() -> None:
    args = parse_args()
    args.results_root = args.results_root.expanduser().resolve()
    args.sequence_root = args.sequence_root.expanduser().resolve()
    rows = read_rows(args.results_root / "raw_results.csv")
    summary = json.loads((args.results_root / "summary.json").read_text(encoding="utf-8"))
    sweep_path = args.results_root / "summary_sweep.csv"
    sweep_rows = read_rows(sweep_path) if sweep_path.exists() else []

    plot_dir = args.results_root / "plots"
    plot_binned(
        rows,
        "dso_confidence",
        plot_dir / "error_vs_dso_confidence.png",
        "DSO confidence",
        np.linspace(0.0, 1.0, 11),
    )
    plot_binned(
        rows,
        "gradient_magnitude",
        plot_dir / "error_vs_gradient_magnitude.png",
        "Gradient magnitude",
        np.linspace(0.0, 0.4, 11),
    )
    plot_binned(
        rows,
        "gt_normal_displacement",
        plot_dir / "error_vs_gt_normal_displacement.png",
        "GT normal displacement",
        np.linspace(-40.0, 40.0, 17),
    )
    for parameter in ["lambda", "sigma", "patch_radius", "orientation_gamma"]:
        plot_sweep(sweep_rows, parameter, plot_dir / f"error_vs_{parameter}.png")
    plot_runtime(summary, plot_dir / "runtime_by_stage.png")
    make_debug_panel(args, rows)
    print(f"Wrote diagnostics under {args.results_root}")


if __name__ == "__main__":
    main()
