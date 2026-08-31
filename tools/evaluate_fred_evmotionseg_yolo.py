#!/usr/bin/env python3
"""Evaluate EvMotionSeg rendered FRED results against FRED Event_YOLO boxes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"frame_(\d+)")


@dataclass
class FrameMetric:
    frame_index: int
    timestamp_us: int
    has_gt: bool
    has_prediction: bool
    gt_box_count: int
    pred_component_count: int
    pred_area_px: int
    prediction_intersection_px: int
    prediction_precision: float
    gt_area_px: int
    best_iou: float
    best_intersection_px: int
    gt_coverage: float
    pred_precision_area: float
    center_error_px: float
    pred_bbox_x1: int
    pred_bbox_y1: int
    pred_bbox_x2: int
    pred_bbox_y2: int
    gt_bbox_x1: int
    gt_bbox_y1: int
    gt_bbox_x2: int
    gt_bbox_y2: int


def event_label_files(label_dir: Path) -> list[Path]:
    files = [path for path in label_dir.glob("Video_*_frame_*.txt") if FRAME_RE.search(path.name)]
    return sorted(files, key=lambda path: int(FRAME_RE.search(path.name).group(1)))


def read_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx_s, cy_s, w_s, h_s = parts
        cx = float(cx_s) * width
        cy = float(cy_s) * height
        bw = float(w_s) * width
        bh = float(h_s) * height
        x1 = max(0, int(np.floor(cx - bw / 2.0)))
        y1 = max(0, int(np.floor(cy - bh / 2.0)))
        x2 = min(width, int(np.ceil(cx + bw / 2.0)))
        y2 = min(height, int(np.ceil(cy + bh / 2.0)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def union_box_mask(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    return mask


def box_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    if not boxes:
        return (-1, -1, -1, -1)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def result_mask(path: Path, white_threshold: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    # EvMotionSeg draws colored points on a white background.
    return np.any(image < white_threshold, axis=2).astype(np.uint8)


def components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[tuple[int, tuple[int, int, int, int], int]]]:
    num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        out.append((label, (x, y, x + w, y + h), area))
    return component_labels, out


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def evaluate_frame(
    frame_index: int,
    timestamp_us: int,
    pred_path: Path,
    label_path: Path,
    width: int,
    height: int,
    min_component_area: int,
    white_threshold: int,
) -> FrameMetric:
    boxes = read_yolo_boxes(label_path, width, height)
    gt_mask = union_box_mask(boxes, width, height)
    gt_area = int(gt_mask.sum())
    gt_bbox = box_union(boxes)

    pred = result_mask(pred_path, white_threshold)
    component_labels, comps = components(pred, min_component_area)
    pred_area = int(sum(area for _, _, area in comps))
    prediction_intersection = 0

    best_iou = 0.0
    best_intersection = 0
    best_coverage = 0.0
    best_precision = 0.0
    best_center_error = float("nan")
    best_bbox = (-1, -1, -1, -1)

    for label, comp_bbox, area in comps:
        x1, y1, x2, y2 = comp_bbox
        component_crop = component_labels[y1:y2, x1:x2] == label
        gt_crop = gt_mask[y1:y2, x1:x2] > 0
        intersection = int(np.count_nonzero(component_crop & gt_crop))
        prediction_intersection += intersection
        if gt_area > 0:
            union = area + gt_area - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_intersection = intersection
                best_coverage = intersection / gt_area if gt_area else 0.0
                best_precision = intersection / area if area else 0.0
                best_bbox = comp_bbox
                pcx, pcy = center(comp_bbox)
                gcx, gcy = center(gt_bbox)
                best_center_error = float(np.hypot(pcx - gcx, pcy - gcy))
    prediction_precision = prediction_intersection / pred_area if pred_area else float("nan")

    return FrameMetric(
        frame_index=frame_index,
        timestamp_us=timestamp_us,
        has_gt=bool(gt_area > 0),
        has_prediction=bool(pred_area > 0),
        gt_box_count=len(boxes),
        pred_component_count=len(comps),
        pred_area_px=pred_area,
        prediction_intersection_px=prediction_intersection,
        prediction_precision=float(prediction_precision),
        gt_area_px=gt_area,
        best_iou=float(best_iou),
        best_intersection_px=best_intersection,
        gt_coverage=float(best_coverage),
        pred_precision_area=float(best_precision),
        center_error_px=float(best_center_error),
        pred_bbox_x1=best_bbox[0],
        pred_bbox_y1=best_bbox[1],
        pred_bbox_x2=best_bbox[2],
        pred_bbox_y2=best_bbox[3],
        gt_bbox_x1=gt_bbox[0],
        gt_bbox_y1=gt_bbox[1],
        gt_bbox_x2=gt_bbox[2],
        gt_bbox_y2=gt_bbox[3],
    )


def missing_prediction_frame(
    frame_index: int,
    timestamp_us: int,
    label_path: Path,
    width: int,
    height: int,
) -> FrameMetric:
    boxes = read_yolo_boxes(label_path, width, height)
    gt_mask = union_box_mask(boxes, width, height)
    gt_area = int(gt_mask.sum())
    gt_bbox = box_union(boxes)
    return FrameMetric(
        frame_index=frame_index,
        timestamp_us=timestamp_us,
        has_gt=bool(gt_area > 0),
        has_prediction=False,
        gt_box_count=len(boxes),
        pred_component_count=0,
        pred_area_px=0,
        prediction_intersection_px=0,
        prediction_precision=float("nan"),
        gt_area_px=gt_area,
        best_iou=0.0,
        best_intersection_px=0,
        gt_coverage=0.0,
        pred_precision_area=0.0,
        center_error_px=float("nan"),
        pred_bbox_x1=-1,
        pred_bbox_y1=-1,
        pred_bbox_x2=-1,
        pred_bbox_y2=-1,
        gt_bbox_x1=gt_bbox[0],
        gt_bbox_y1=gt_bbox[1],
        gt_bbox_x2=gt_bbox[2],
        gt_bbox_y2=gt_bbox[3],
    )


def summarize(rows: list[FrameMetric], iou_threshold: float, coverage_threshold: float) -> dict:
    positives = [row for row in rows if row.has_gt]
    negatives = [row for row in rows if not row.has_gt]
    predicted = [row for row in rows if row.has_prediction]
    precision_values = [row.prediction_precision for row in predicted if np.isfinite(row.prediction_precision)]
    detected_iou = [row for row in positives if row.best_iou >= iou_threshold]
    detected_cov = [row for row in positives if row.gt_coverage >= coverage_threshold]
    false_positive_frames = [row for row in negatives if row.has_prediction]
    center_errors = np.array([row.center_error_px for row in positives if np.isfinite(row.center_error_px)])

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "frames_total": len(rows),
        "frames_with_gt": len(positives),
        "frames_without_gt": len(negatives),
        "frames_with_prediction": int(sum(row.has_prediction for row in rows)),
        "prediction_frame_rate": len(predicted) / len(rows) if rows else float("nan"),
        "iou_threshold": iou_threshold,
        "coverage_threshold": coverage_threshold,
        "box_hit_rate_iou": len(detected_iou) / len(positives) if positives else float("nan"),
        "box_hit_rate_coverage": len(detected_cov) / len(positives) if positives else float("nan"),
        "false_positive_frame_rate": len(false_positive_frames) / len(negatives) if negatives else float("nan"),
        "mean_best_iou_positive_frames": mean([row.best_iou for row in positives]),
        "median_best_iou_positive_frames": float(np.median([row.best_iou for row in positives])) if positives else float("nan"),
        "mean_gt_coverage_positive_frames": mean([row.gt_coverage for row in positives]),
        "mean_pred_precision_area_positive_frames": mean([row.pred_precision_area for row in positives]),
        "mean_prediction_precision_predicted_frames": mean(precision_values),
        "median_prediction_precision_predicted_frames": float(np.median(precision_values)) if precision_values else float("nan"),
        "mean_prediction_precision_all_frames_zero_for_no_prediction": mean(
            [row.prediction_precision if np.isfinite(row.prediction_precision) else 0.0 for row in rows]
        ),
        "precision_hit_rate_0_25_predicted_frames": (
            sum(value >= 0.25 for value in precision_values) / len(precision_values) if precision_values else float("nan")
        ),
        "precision_hit_rate_0_50_predicted_frames": (
            sum(value >= 0.50 for value in precision_values) / len(precision_values) if precision_values else float("nan")
        ),
        "precision_hit_rate_0_75_predicted_frames": (
            sum(value >= 0.75 for value in precision_values) / len(precision_values) if precision_values else float("nan")
        ),
        "precision_hit_rate_0_90_predicted_frames": (
            sum(value >= 0.90 for value in precision_values) / len(precision_values) if precision_values else float("nan")
        ),
        "mean_center_error_px_detected": float(center_errors.mean()) if center_errors.size else float("nan"),
        "median_center_error_px_detected": float(np.median(center_errors)) if center_errors.size else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--event-yolo-dir", type=Path, default=Path("Datasets/FRED/0/Event_YOLO"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-component-area", type=int, default=2)
    parser.add_argument("--white-threshold", type=int, default=250)
    parser.add_argument("--iou-threshold", type=float, default=0.01)
    parser.add_argument("--coverage-threshold", type=float, default=0.05)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--prediction-frame-offset",
        type=int,
        default=0,
        help="Compare YOLO label frame i against prediction frame i + offset.",
    )
    parser.add_argument("--gt-only", action="store_true", help="Only evaluate frames with at least one YOLO box.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_files = event_label_files(args.event_yolo_dir)
    rows: list[FrameMetric] = []
    missing_results = []
    for frame_index, label_file in enumerate(label_files):
        if frame_index < args.start_frame:
            continue
        if args.end_frame is not None and frame_index > args.end_frame:
            continue
        boxes = read_yolo_boxes(label_file, args.width, args.height)
        if args.gt_only and not boxes:
            continue
        prediction_frame_index = frame_index + args.prediction_frame_offset
        result_path = args.results_dir / f"{prediction_frame_index}.png"
        timestamp_us = int(FRAME_RE.search(label_file.name).group(1))
        if not result_path.exists():
            missing_results.append(frame_index)
            rows.append(
                missing_prediction_frame(
                    frame_index,
                    timestamp_us,
                    label_file,
                    args.width,
                    args.height,
                )
            )
            continue
        rows.append(
            evaluate_frame(
                frame_index,
                timestamp_us,
                result_path,
                label_file,
                args.width,
                args.height,
                args.min_component_area,
                args.white_threshold,
            )
        )

    csv_path = args.output_dir / "frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(FrameMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary = summarize(rows, args.iou_threshold, args.coverage_threshold) if rows else {}
    summary.update(
        {
            "results_dir": str(args.results_dir),
            "event_yolo_dir": str(args.event_yolo_dir),
            "output_dir": str(args.output_dir),
            "evaluated_frames": len(rows),
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "prediction_frame_offset": args.prediction_frame_offset,
            "gt_only": args.gt_only,
            "missing_result_frames": missing_results[:100],
            "missing_result_frame_count": len(missing_results),
            "metric_note": "EvMotionSeg result images are rendered colored points on white background; prediction masks are non-white pixels. YOLO boxes are weak labels, not pixel masks.",
        }
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
