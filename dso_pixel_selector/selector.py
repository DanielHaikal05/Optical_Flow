#!/usr/bin/env python3
"""Small Python mirror of DSO's frontend pixel-selection idea.

This is intentionally limited to point selection: no pose, depth, tracking, or
bundle adjustment is used. It follows the important DSO selector ingredients:
local gradient histograms, smoothed adaptive thresholds, deterministic
directional selection, and multi-size block picks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DSOSelectorConfig:
    potential: int = 3
    hist_cell_size: int = 32
    min_grad_hist_cut: float = 0.5
    min_grad_hist_add: float = 7.0 / 255.0
    grad_downweight_per_level: float = 0.75
    th_factor: float = 1.0
    border: int = 4
    dso_sigma: float = 4.0
    use_direction_distribution: bool = True


DSO_DIRECTIONS = np.asarray(
    [
        [0.0, 1.0],
        [0.3827, 0.9239],
        [0.1951, 0.9808],
        [0.9239, 0.3827],
        [0.7071, 0.7071],
        [0.3827, -0.9239],
        [0.8315, 0.5556],
        [0.8315, -0.5556],
        [0.5556, -0.8315],
        [0.9808, 0.1951],
        [0.9239, -0.3827],
        [0.7071, -0.7071],
        [0.5556, 0.8315],
        [0.9808, -0.1951],
        [1.0, 0.0],
        [0.1951, -0.9808],
    ],
    dtype=np.float32,
)


def _gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = cv2.Scharr(image, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    iy = cv2.Scharr(image, cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    grad_sq = ix * ix + iy * iy
    return ix, iy, grad_sq


def _local_thresholds(
    grad_mag: np.ndarray,
    config: DSOSelectorConfig,
) -> np.ndarray:
    h, w = grad_mag.shape
    cell = config.hist_cell_size
    ny = max(1, int(np.ceil(h / cell)))
    nx = max(1, int(np.ceil(w / cell)))

    thresholds = np.zeros((ny, nx), dtype=np.float32)

    for cy in range(ny):
        for cx in range(nx):
            y0 = cy * cell
            x0 = cx * cell
            patch = grad_mag[y0 : min(y0 + cell, h), x0 : min(x0 + cell, w)]
            patch = patch[np.isfinite(patch)]
            if patch.size == 0:
                thresholds[cy, cx] = config.min_grad_hist_add
            else:
                thresholds[cy, cx] = (
                    np.quantile(patch, config.min_grad_hist_cut)
                    + config.min_grad_hist_add
                )

    smoothed = cv2.blur(thresholds, (3, 3), borderType=cv2.BORDER_REPLICATE)
    return smoothed


def _threshold_at(thresholds: np.ndarray, x: int, y: int, cell: int) -> float:
    cy = min(thresholds.shape[0] - 1, max(0, y // cell))
    cx = min(thresholds.shape[1] - 1, max(0, x // cell))
    return float(thresholds[cy, cx])


def _best_in_region(
    ix: np.ndarray,
    iy: np.ndarray,
    grad_sq: np.ndarray,
    thresholds: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    direction: np.ndarray,
    min_threshold_scale: float,
    config: DSOSelectorConfig,
) -> tuple[int, int] | None:
    h, w = grad_sq.shape
    best_xy = None
    best_score = -np.inf

    x0 = max(config.border, x0)
    y0 = max(config.border, y0)
    x1 = min(w - config.border - 1, x1)
    y1 = min(h - config.border - 1, y1)

    if x0 >= x1 or y0 >= y1:
        return None

    region_grad_sq = grad_sq[y0:y1, x0:x1]
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    cell_y = np.minimum(thresholds.shape[0] - 1, ys // config.hist_cell_size)
    cell_x = np.minimum(thresholds.shape[1] - 1, xs // config.hist_cell_size)
    local_thresholds = thresholds[cell_y, cell_x]
    valid = region_grad_sq > (
        local_thresholds
        * local_thresholds
        * min_threshold_scale
        * config.th_factor
    )

    if not np.any(valid):
        return None

    if config.use_direction_distribution:
        score = np.abs(
            ix[y0:y1, x0:x1] * direction[0]
            + iy[y0:y1, x0:x1] * direction[1]
        )
    else:
        score = region_grad_sq

    score = np.where(valid, score, -np.inf)
    rel_y, rel_x = np.unravel_index(np.argmax(score), score.shape)
    best_score = float(score[rel_y, rel_x])
    if not np.isfinite(best_score):
        return None
    best_xy = (x0 + int(rel_x), y0 + int(rel_y))

    return best_xy


def select_dso_points(
    image: np.ndarray,
    config: DSOSelectorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a DSO-style binary mask and Nx2 `(x, y)` points array."""

    if config is None:
        config = DSOSelectorConfig()

    image = image.astype(np.float32)
    ix, iy, grad_sq0 = _gradient(image)
    grad_mag0 = np.sqrt(grad_sq0)

    grad_sq1 = cv2.pyrDown(grad_sq0)
    grad_sq2 = cv2.pyrDown(grad_sq1)
    thresholds = _local_thresholds(grad_mag0, config)

    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    points: list[tuple[int, int]] = []
    rng = np.random.default_rng(3141592)
    direction_ids = rng.integers(0, len(DSO_DIRECTIONS), size=max(1, h * w // 8))
    dir_cursor = 0

    pot = max(1, int(config.potential))
    dw1 = config.grad_downweight_per_level
    dw2 = dw1 * dw1

    for y4 in range(0, h, 4 * pot):
        for x4 in range(0, w, 4 * pot):
            dir4 = DSO_DIRECTIONS[direction_ids[dir_cursor % len(direction_ids)]]
            dir_cursor += 1
            best4 = _best_in_region(
                ix,
                iy,
                grad_sq0,
                thresholds,
                x4,
                y4,
                min(x4 + 4 * pot, w),
                min(y4 + 4 * pot, h),
                dir4,
                dw2,
                config,
            )

            for y3 in range(y4, min(y4 + 4 * pot, h), 2 * pot):
                for x3 in range(x4, min(x4 + 4 * pot, w), 2 * pot):
                    dir3 = DSO_DIRECTIONS[direction_ids[dir_cursor % len(direction_ids)]]
                    dir_cursor += 1
                    best3 = _best_in_region(
                        ix,
                        iy,
                        grad_sq0,
                        thresholds,
                        x3,
                        y3,
                        min(x3 + 2 * pot, w),
                        min(y3 + 2 * pot, h),
                        dir3,
                        dw1,
                        config,
                    )

                    for y2 in range(y3, min(y3 + 2 * pot, h), pot):
                        for x2 in range(x3, min(x3 + 2 * pot, w), pot):
                            dir2 = DSO_DIRECTIONS[
                                direction_ids[dir_cursor % len(direction_ids)]
                            ]
                            dir_cursor += 1
                            best2 = _best_in_region(
                                ix,
                                iy,
                                grad_sq0,
                                thresholds,
                                x2,
                                y2,
                                min(x2 + pot, w),
                                min(y2 + pot, h),
                                dir2,
                                1.0,
                                config,
                            )
                            if best2 is not None:
                                points.append(best2)
                                mask[best2[1], best2[0]] = 255
                                best3 = None
                                best4 = None

                    if best3 is not None:
                        points.append(best3)
                        mask[best3[1], best3[0]] = 255
                        best4 = None

            if best4 is not None:
                points.append(best4)
                mask[best4[1], best4[0]] = 255

    if not points:
        return mask, np.empty((0, 2), dtype=np.int32)

    unique_points = np.unique(np.asarray(points, dtype=np.int32), axis=0)
    mask.fill(0)
    mask[unique_points[:, 1], unique_points[:, 0]] = 255
    return mask, unique_points


def build_dso_confidence(
    dso_mask: np.ndarray,
    sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build max-Gaussian confidence and nearest-point distance fields."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    binary = (dso_mask > 0).astype(np.uint8)
    if not np.any(binary):
        distance = np.full(dso_mask.shape, np.inf, dtype=np.float32)
        confidence = np.zeros(dso_mask.shape, dtype=np.float32)
        return confidence, distance

    distance = cv2.distanceTransform(
        1 - binary,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    ).astype(np.float32)
    confidence = np.exp(
        -(distance * distance)
        /
        (2.0 * sigma * sigma)
    ).astype(np.float32)
    confidence[binary > 0] = 1.0
    return confidence, distance


def draw_selected_points(
    image_bgr: np.ndarray,
    dso_mask: np.ndarray,
    output_path: str | Path,
) -> None:
    output = image_bgr.copy()
    ys, xs = np.nonzero(dso_mask > 0)
    for x, y in zip(xs, ys):
        cv2.circle(output, (int(x), int(y)), 1, (0, 255, 0), -1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)


def save_confidence_heatmap(
    confidence: np.ndarray,
    output_path: str | Path,
) -> None:
    heat = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), heat)
