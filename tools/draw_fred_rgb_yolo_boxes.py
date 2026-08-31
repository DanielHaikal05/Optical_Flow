#!/usr/bin/env python3
"""Draw FRED RGB_YOLO boxes over matching RGB frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    boxes: list[tuple[int, int, int, int, int]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id_s, cx_s, cy_s, w_s, h_s = parts
        cx = float(cx_s) * width
        cy = float(cy_s) * height
        bw = float(w_s) * width
        bh = float(h_s) * height
        x1 = max(0, int(np.floor(cx - bw / 2.0)))
        y1 = max(0, int(np.floor(cy - bh / 2.0)))
        x2 = min(width - 1, int(np.ceil(cx + bw / 2.0)))
        y2 = min(height - 1, int(np.ceil(cy + bh / 2.0)))
        if x2 > x1 and y2 > y1:
            boxes.append((int(class_id_s), x1, y1, x2, y2))
    return boxes


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int, int]], frame_name: str) -> np.ndarray:
    out = image.copy()
    color = (0, 255, 255)
    for class_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"class {class_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_text), (min(out.shape[1] - 1, x1 + tw + 8), y_text + th + baseline + 6), color, -1)
        cv2.putText(out, label, (x1 + 4, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(out, frame_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, frame_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=Path("Datasets/FRED/0/RGB"))
    parser.add_argument("--label-dir", type=Path, default=Path("Datasets/FRED/0/RGB_YOLO"))
    parser.add_argument("--output-dir", type=Path, default=Path("Datasets/FRED/0/RGB_YOLO_BOX_OVERLAYS_GT"))
    parser.add_argument("--include-empty", action="store_true", help="Also write frames with no YOLO boxes.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.rgb_dir.glob("*.jpg"))
    written = 0
    skipped_empty = 0
    missing_labels = 0
    failed_images = 0

    for image_path in images:
        label_path = args.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failed_images += 1
            continue
        height, width = image.shape[:2]
        boxes = read_yolo_boxes(label_path, width, height)
        if not boxes and not args.include_empty:
            skipped_empty += 1
            continue
        overlay = draw_boxes(image, boxes, image_path.stem)
        out_path = args.output_dir / image_path.name
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        written += 1

    summary = {
        "rgb_dir": str(args.rgb_dir),
        "label_dir": str(args.label_dir),
        "output_dir": str(args.output_dir),
        "rgb_frames": len(images),
        "written": written,
        "skipped_empty": skipped_empty,
        "missing_labels": missing_labels,
        "failed_images": failed_images,
        "include_empty": args.include_empty,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
