#!/usr/bin/env python3
"""DSO-style photometric calibration helpers for grayscale/RGB images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class PhotometricCalibrator:
    """
    Apply inverse response, vignette, and exposure correction.

    The corrected image is proportional to irradiance:

        B(x) = G^-1(I_raw(x)) / (exposure_time * vignette(x))
    """

    def __init__(
        self,
        response_curve=None,
        vignette=None,
        use_exposure=True,
        vignette_min=1e-3,
        output_scale=1.0,
    ):
        self.response_curve = (
            None
            if response_curve is None
            else self._normalize_response_curve(response_curve)
        )
        self.vignette = (
            None
            if vignette is None
            else self._normalize_vignette(vignette)
        )
        self.use_exposure = use_exposure
        self.vignette_min = float(vignette_min)
        self.output_scale = float(output_scale)

    @classmethod
    def from_files(
        cls,
        response_path=None,
        vignette_path=None,
        **kwargs,
    ):
        response_curve = (
            None
            if response_path is None
            else load_response_curve(response_path)
        )
        vignette = (
            None
            if vignette_path is None
            else load_vignette(vignette_path)
        )
        return cls(
            response_curve=response_curve,
            vignette=vignette,
            **kwargs,
        )

    def correct(
        self,
        image,
        exposure_time=None,
    ):
        gray = as_float_gray(image)

        if self.response_curve is not None:
            gray = apply_response_curve(
                gray,
                self.response_curve
            )

        if self.vignette is not None:
            if self.vignette.shape != gray.shape:
                raise ValueError(
                    f"Vignette shape {self.vignette.shape} does not match image shape {gray.shape}"
                )

            valid = self.vignette > self.vignette_min
            corrected = np.zeros_like(gray, dtype=np.float32)
            corrected[valid] = (
                gray[valid]
                /
                (self.vignette[valid] + 1e-8)
            )
            gray = corrected

        if self.use_exposure and exposure_time is not None:
            exposure = max(float(exposure_time), 1e-12)
            gray = gray / exposure

        return (
            gray * self.output_scale
        ).astype(np.float32)

    def valid_mask(self, shape):
        if self.vignette is None:
            return np.ones(shape, dtype=bool)
        if self.vignette.shape != shape:
            raise ValueError(
                f"Vignette shape {self.vignette.shape} does not match requested shape {shape}"
            )
        return self.vignette > self.vignette_min

    @staticmethod
    def _normalize_response_curve(response_curve):
        response = np.asarray(response_curve, dtype=np.float32).reshape(-1)
        if response.size != 256:
            raise ValueError(
                f"Response curve must contain 256 values, got {response.size}"
            )
        max_value = float(np.max(response))
        if max_value > 0:
            response = response / max_value
        return response.astype(np.float32)

    @staticmethod
    def _normalize_vignette(vignette):
        vignette = np.asarray(vignette, dtype=np.float32)
        max_value = float(np.max(vignette))
        if max_value <= 0:
            raise ValueError("Vignette must contain positive values")
        vignette = vignette / max_value
        return vignette.astype(np.float32)


def as_float_gray(image):
    arr = np.asarray(image)

    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError(
            f"Expected a grayscale or BGR image, got shape {arr.shape}"
        )

    gray = gray.astype(np.float32)
    if gray.dtype == np.uint8 or np.max(gray) > 1.5:
        gray = gray / 255.0

    return gray.astype(np.float32)


def load_response_curve(path):
    values = np.loadtxt(
        Path(path).expanduser(),
        dtype=np.float32
    ).reshape(-1)
    if values.size != 256:
        raise ValueError(
            f"Response file must contain 256 values, got {values.size}: {path}"
        )
    return values


def load_vignette(path):
    image = cv2.imread(
        str(Path(path).expanduser()),
        cv2.IMREAD_UNCHANGED
    )
    if image is None:
        raise FileNotFoundError(path)
    return as_float_gray(image)


def apply_response_curve(image, response_curve):
    gray = as_float_gray(image)
    response = PhotometricCalibrator._normalize_response_curve(
        response_curve
    )

    xs = np.linspace(
        0.0,
        1.0,
        response.size,
        dtype=np.float32
    )

    return np.interp(
        gray.reshape(-1),
        xs,
        response
    ).reshape(gray.shape).astype(np.float32)


def make_gamma_inverse_response(gamma):
    raw = np.linspace(
        0.0,
        1.0,
        256,
        dtype=np.float32
    )
    return np.power(
        raw,
        float(gamma)
    ).astype(np.float32)


def make_radial_vignette(
    shape,
    k1=0.25,
    k2=0.15,
):
    height, width = shape
    yy, xx = np.mgrid[
        0:height,
        0:width
    ].astype(np.float32)

    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    rx = (xx - cx) / max(cx, 1.0)
    ry = (yy - cy) / max(cy, 1.0)
    r = np.sqrt(
        rx * rx + ry * ry
    ) / np.sqrt(2.0)

    vignette = (
        1.0
        - k1 * r**2
        - k2 * r**4
    )

    return np.clip(
        vignette,
        0.05,
        1.0
    ).astype(np.float32)


def apply_synthetic_photometric_model(
    linear_image,
    exposure_time=1.0,
    gamma=2.2,
    vignette=None,
):
    """
    Simulate I_raw = G(t * V * B), with G(x)=x^(1/gamma).
    """

    image = as_float_gray(linear_image)
    scaled = image.astype(np.float32) * float(exposure_time)

    if vignette is not None:
        scaled = scaled * np.asarray(vignette, dtype=np.float32)

    scaled = np.clip(
        scaled,
        0.0,
        1.0
    )

    return np.power(
        scaled,
        1.0 / float(gamma)
    ).astype(np.float32)
