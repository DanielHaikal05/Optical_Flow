#!/usr/bin/env python3
"""Create qualitative EvMotionSeg/FRED overlays for debugging."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"frame_(\d+)")


def sorted_event_files(path: Path, suffix: str) -> list[Path]:
    files = [p for p in path.glob(f"Video_*_frame_*.{suffix}") if FRAME_RE.search(p.name)]
    return sorted(files, key=lambda p: int(FRAME_RE.search(p.name).group(1)))


def read_yolo(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx_s, cy_s, w_s, h_s = parts
        cx = float(cx_s) * width
        cy = float(cy_s) * height
        bw = float(w_s) * width
        bh = float(h_s) * height
        x1 = max(0, int(np.floor(cx - bw / 2)))
        y1 = max(0, int(np.floor(cy - bh / 2)))
        x2 = min(width - 1, int(np.ceil(cx + bw / 2)))
        y2 = min(height - 1, int(np.ceil(cy + bh / 2)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def read_metrics(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row["frame_index"]): row for row in csv.DictReader(f)}


def metrics_csv_path(metrics_dir: Path | None, evmotion_root: Path, seq: str) -> Path:
    if metrics_dir is None:
        return (
            evmotion_root
            / f"fred_{seq}_sample2000"
            / "evaluation_yolo_imo_from700_gt_only_precision"
            / "frame_metrics.csv"
        )
    direct = metrics_dir / "frame_metrics.csv"
    if direct.exists():
        return direct
    return metrics_dir / f"seq_{seq}" / "frame_metrics.csv"


def nonwhite_mask(path: Path, white_threshold: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((0, 0), dtype=bool)
    return np.any(image < white_threshold, axis=2)


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]], color: tuple[int, int, int]) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_panel(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_overlay(
    seq: str,
    frame: int,
    prediction_frame_offset: int,
    overlay_source: str,
    fred_root: Path,
    evmotion_root: Path,
    metrics: dict[int, dict[str, str]],
    output_dir: Path,
    run_dir: Path | None,
    width: int,
    height: int,
    white_threshold: int,
) -> Path | None:
    event_files = sorted_event_files(fred_root / seq / "Event" / "Frames", "png")
    label_files = sorted_event_files(fred_root / seq / "Event_YOLO", "txt")
    if frame >= len(event_files) or frame >= len(label_files):
        return None

    event_image = cv2.imread(str(event_files[frame]), cv2.IMREAD_COLOR)
    if event_image is None:
        return None
    if event_image.shape[:2] != (height, width):
        event_image = cv2.resize(event_image, (width, height), interpolation=cv2.INTER_AREA)

    timestamp = int(FRAME_RE.search(event_files[frame].name).group(1))
    boxes = read_yolo(label_files[frame], width, height)
    prediction_frame = frame + prediction_frame_offset

    sequence_dir = run_dir if run_dir else evmotion_root / f"fred_{seq}_sample2000"
    result_path = sequence_dir / "results" / f"{prediction_frame}.png"
    imo_path = sequence_dir / "results_imo" / f"{prediction_frame}.png"
    result = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
    imo = cv2.imread(str(imo_path), cv2.IMREAD_COLOR)
    if result is None:
        result = np.full_like(event_image, 255)
    if imo is None:
        imo = np.full_like(event_image, 255)

    event_gt = event_image.copy()
    draw_boxes(event_gt, boxes, (0, 0, 255))

    mask = np.zeros((height, width), dtype=bool)
    overlay_mask_path = result_path if overlay_source == "all" else imo_path
    raw_mask = nonwhite_mask(overlay_mask_path, white_threshold)
    if raw_mask.size:
        mask = raw_mask
    overlay = event_gt.copy()
    overlay[mask] = (255, 255, 0)
    overlay = cv2.addWeighted(event_gt, 0.72, overlay, 0.28, 0.0)
    draw_boxes(overlay, boxes, (0, 0, 255))

    metric = metrics.get(frame, {})
    precision = metric.get("prediction_precision", "nan")
    pred_area = metric.get("pred_area_px", "?")
    intersection = metric.get("prediction_intersection_px", "?")
    gt_area = metric.get("gt_area_px", "?")

    panels = [
        add_label(event_gt, f"seq {seq} frame {frame} event+YOLO ts={timestamp}"),
        add_label(result, f"EvMotionSeg all labels: prediction frame {prediction_frame}"),
        add_label(imo, f"IMO-only mask: prediction frame {prediction_frame}"),
        add_label(
            overlay,
            f"cyan={overlay_source} red=YOLO precision={precision} pred={pred_area} hit={intersection} gt={gt_area}",
        ),
    ]
    panels = [fit_panel(panel, 640) for panel in panels]
    sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fred_{seq}_frame_{frame:04d}.jpg"
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", action="append", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--fred-root", type=Path, default=Path("Datasets/FRED"))
    parser.add_argument("--evmotion-root", type=Path, default=Path("EvMotionSeg/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/fred_evmotionseg_debug/qualitative"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--white-threshold", type=int, default=250)
    parser.add_argument(
        "--prediction-frame-offset",
        type=int,
        default=0,
        help="Compare YOLO label frame i against EvMotionSeg prediction frame i + offset.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Optional directory containing frame_metrics.csv. Defaults to the original zero-offset metrics dir.",
    )
    parser.add_argument("--overlay-source", choices=("imo", "all"), default="imo")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional EvMotionSeg sequence directory with results/ and results_imo/. Use with one --seq.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir and len(args.seq) != 1:
        raise ValueError("--run-dir can only be used with exactly one --seq")
    made = []
    for seq in args.seq:
        metrics = read_metrics(metrics_csv_path(args.metrics_dir, args.evmotion_root, seq))
        for frame in args.frames:
            out_path = make_overlay(
                seq=seq,
                frame=frame,
                prediction_frame_offset=args.prediction_frame_offset,
                overlay_source=args.overlay_source,
                fred_root=args.fred_root,
                evmotion_root=args.evmotion_root,
                metrics=metrics,
                output_dir=args.output_dir / f"seq_{seq}",
                run_dir=args.run_dir,
                width=args.width,
                height=args.height,
                white_threshold=args.white_threshold,
            )
            if out_path:
                made.append(str(out_path))
    print("\n".join(made))


if __name__ == "__main__":
    main()
