#!/usr/bin/env python3
"""Sanity tests for DSO-weighted patch normal flow."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Camera_NF import (  # noqa: E402
    compute_normal_flow_patch,
    compute_normal_flow_patch_dso,
    preprocess,
)


def synthetic_pair(shift_x: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = 96, 128
    x = np.clip(np.arange(w, dtype=np.float32) / 32.0, 0.0, 1.0)
    image = np.tile(x[None, :], (h, 1))
    transform = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, 0.0]])
    shifted = cv2.warpAffine(
        image,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return preprocess(image), preprocess(shifted)


def test_dso_lambda_zero_matches_patch() -> None:
    image0, image1 = synthetic_pair()
    locations = [(48, x) for x in range(24, 104, 8)]
    confidence = np.ones_like(image0, dtype=np.float32)

    patch = compute_normal_flow_patch(
        image0,
        image0,
        image1,
        0.0,
        0.0,
        1.0,
        locations=locations,
    )
    dso_zero = compute_normal_flow_patch_dso(
        image0,
        image0,
        image1,
        0.0,
        0.0,
        1.0,
        confidence,
        locations=locations,
        lambda_dso=0.0,
    )

    np.testing.assert_allclose(patch[0], dso_zero[0], atol=1e-7)
    np.testing.assert_allclose(patch[1], dso_zero[1], atol=1e-7)
    np.testing.assert_array_equal(patch[3], dso_zero[3])


def test_synthetic_translation_recovers_normal_displacement() -> None:
    image0, image1 = synthetic_pair(shift_x=2.0)
    locations = [(48, x) for x in range(8, 32, 4)]

    vx, vy, _mag, valid, *_rest = compute_normal_flow_patch(
        image0,
        image0,
        image1,
        0.0,
        0.0,
        1.0,
        locations=locations,
    )

    estimates = np.asarray([vx[y, x] for y, x in locations if valid[y, x]])
    assert estimates.size > 0
    assert abs(float(np.median(estimates)) - 2.0) < 0.25
    assert np.max(np.abs([vy[y, x] for y, x in locations if valid[y, x]])) < 0.25
