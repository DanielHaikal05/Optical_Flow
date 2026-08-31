#!/usr/bin/env python3

import cv2
import numpy as np
import time


# ============================================================
# Settings
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Gaussian smoothing before differentiation
GAUSSIAN_SIGMA = 1.2

# Reject pixels with weak spatial gradients
GRAD_THRESHOLD = 0.02

# Divide the image into cells of this size
GRID_STEP = 20

# Arrow visualization scale
# This ONLY affects how long the arrows look on screen
ARROW_SCALE = 0.10

# Maximum displayed arrow length relative to grid size
MAX_ARROW_LENGTH_FACTOR = 1.5

# Patch-based direct normal-displacement settings
PATCH_RADIUS = 2
MAX_DISPLACEMENT = 8.0
COARSE_STEP = 0.5
FINE_STEP = 0.1
HUBER_DELTA = 0.03
MIN_PATCH_SAMPLES = 8
DSO_SIGMA = 4.0
DSO_WEIGHT_LAMBDA = 2.0
ORIENTATION_GAMMA = 2.0
PATCH_SPATIAL_SIGMA = 2.0

# Robust global affine-brightness fitting settings
AFFINE_MIN_INTENSITY = 0.03
AFFINE_MAX_INTENSITY = 0.97
AFFINE_OUTLIER_MAD_SCALE = 3.0


# ============================================================
# Image preprocessing
# ============================================================

def _to_grayscale_float(frame):
    """
    Convert BGR or grayscale input to float grayscale.
    """

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(
            f"Unsupported frame shape: {frame.shape}"
        )

    gray = gray.astype(np.float32)

    if np.max(gray) > 1.5:
        gray = gray / 255.0

    return gray


def preprocess(frame, calibrator=None, exposure_time=None):
    """
    Convert camera image to smoothed grayscale float image.

    Output intensity range:
        [0, 1] for the baseline path. Photometrically calibrated
        images keep a fixed irradiance-like scale.
    """

    if calibrator is None:
        gray = _to_grayscale_float(frame)
    else:
        gray = calibrator.correct(
            frame,
            exposure_time=exposure_time
        )

    gray = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA
    )

    return gray


# ============================================================
# Shared normal-flow helpers
# ============================================================

def compute_image_derivatives(I_curr):
    """
    Compute Scharr spatial derivatives used by all variants.
    """

    Ix = cv2.Scharr(
        I_curr,
        cv2.CV_32F,
        1,
        0,
        scale=1.0 / 32.0
    )

    Iy = cv2.Scharr(
        I_curr,
        cv2.CV_32F,
        0,
        1,
        scale=1.0 / 32.0
    )

    return Ix, Iy


def solve_normal_flow_from_derivatives(Ix, Iy, It):
    """
    Analytically solve normal flow from image derivatives.
    """

    grad_sq = Ix**2 + Iy**2

    grad_mag = np.sqrt(grad_sq)

    valid = grad_mag > GRAD_THRESHOLD

    vx = np.zeros_like(Ix)
    vy = np.zeros_like(Iy)

    eps = 1e-8

    factor = np.zeros_like(Ix)

    factor[valid] = (
        -It[valid]
        /
        (grad_sq[valid] + eps)
    )

    vx[valid] = factor[valid] * Ix[valid]
    vy[valid] = factor[valid] * Iy[valid]

    magnitude = np.sqrt(
        vx**2 + vy**2
    )

    return vx, vy, magnitude, valid


def estimate_affine_brightness(reference, target, mask=None):
    """
    Estimate target ~= a * reference + b using a robust global fit.

    Saturated and very dark samples are ignored. A second fit rejects
    residual outliers using a MAD threshold.
    """

    if reference.shape != target.shape:
        raise ValueError(
            "reference and target must have the same shape"
        )

    fit_mask = (
        np.isfinite(reference)
        & np.isfinite(target)
        & (reference > AFFINE_MIN_INTENSITY)
        & (target > AFFINE_MIN_INTENSITY)
        & (reference < AFFINE_MAX_INTENSITY)
        & (target < AFFINE_MAX_INTENSITY)
    )

    if mask is not None:
        fit_mask &= mask

    x = reference[fit_mask].reshape(-1)
    y = target[fit_mask].reshape(-1)

    if x.size < 2:
        return 1.0, 0.0

    A = np.column_stack(
        (x, np.ones_like(x))
    )

    a, b = np.linalg.lstsq(
        A,
        y,
        rcond=None
    )[0]

    residual = y - (a * x + b)
    median = np.median(residual)
    mad = np.median(
        np.abs(residual - median)
    )

    robust_sigma = 1.4826 * mad

    if robust_sigma > 1e-6:
        inlier = (
            np.abs(residual - median)
            <= AFFINE_OUTLIER_MAD_SCALE * robust_sigma
        )

        if np.count_nonzero(inlier) >= 2:
            A_in = A[inlier]
            y_in = y[inlier]

            a, b = np.linalg.lstsq(
                A_in,
                y_in,
                rcond=None
            )[0]

    return float(a), float(b)


def apply_affine_brightness(image, a, b):
    """
    Apply an affine brightness mapping and keep intensities in [0, 1].
    """

    return np.clip(
        a * image + b,
        0.0,
        1.0
    ).astype(np.float32)


def select_strongest_gradient_points(
    valid,
    strength,
    step=GRID_STEP
):
    """
    Select one valid point per grid cell using maximum gradient strength.

    Returns a list of (y, x) integer coordinates.
    """

    height, width = valid.shape
    points = []

    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            y1 = min(y0 + step, height)
            x1 = min(x0 + step, width)

            cell_valid = valid[y0:y1, x0:x1]

            if not np.any(cell_valid):
                continue

            cell_strength = strength[y0:y1, x0:x1].copy()
            cell_strength[~cell_valid] = -1.0

            relative_y, relative_x = np.unravel_index(
                np.argmax(cell_strength),
                cell_strength.shape
            )

            points.append(
                (y0 + relative_y, x0 + relative_x)
            )

    return points


def _sample_bilinear(image, xs, ys):
    """
    Vectorized bilinear sampling for one grayscale image.
    """

    height, width = image.shape

    valid = (
        (xs >= 0.0)
        & (ys >= 0.0)
        & (xs < width - 1)
        & (ys < height - 1)
    )

    values = np.zeros_like(xs, dtype=np.float32)

    if not np.any(valid):
        return values, valid

    xv = xs[valid]
    yv = ys[valid]

    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = xv - x0
    wy = yv - y0

    top = (
        (1.0 - wx) * image[y0, x0]
        + wx * image[y0, x1]
    )
    bottom = (
        (1.0 - wx) * image[y1, x0]
        + wx * image[y1, x1]
    )

    values[valid] = (
        (1.0 - wy) * top
        + wy * bottom
    )

    return values, valid


def _huber_mean(residuals, delta=HUBER_DELTA):
    """
    Mean Huber penalty for photometric residuals.
    """

    abs_residuals = np.abs(residuals)
    quadratic = abs_residuals <= delta

    losses = np.empty_like(residuals, dtype=np.float32)
    losses[quadratic] = 0.5 * residuals[quadratic] ** 2
    losses[~quadratic] = (
        delta * (abs_residuals[~quadratic] - 0.5 * delta)
    )

    return float(np.mean(losses))


def _weighted_huber_mean(residuals, weights=None, delta=HUBER_DELTA):
    """
    Weighted mean Huber penalty for photometric residuals.
    """

    abs_residuals = np.abs(residuals)
    quadratic = abs_residuals <= delta

    losses = np.empty_like(residuals, dtype=np.float32)
    losses[quadratic] = 0.5 * residuals[quadratic] ** 2
    losses[~quadratic] = (
        delta * (abs_residuals[~quadratic] - 0.5 * delta)
    )

    if weights is None:
        return float(np.mean(losses))

    weights = weights.astype(np.float32)
    weight_sum = float(np.sum(weights))

    if weight_sum <= 1e-8:
        return np.inf

    return float(
        np.sum(weights * losses)
        /
        weight_sum
    )


def estimate_patch_normal_displacement(
    image_ref,
    image_target,
    x,
    y,
    nx,
    ny,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP,
    huber_delta=HUBER_DELTA,
    min_samples=MIN_PATCH_SAMPLES
):
    """
    Estimate a scalar displacement along one normal direction for a patch.

    The patch supplies multiple photometric residuals while the unknown is
    only the scalar normal displacement d_n.
    """

    return estimate_patch_normal_displacement_weighted(
        image_ref,
        image_target,
        x,
        y,
        nx,
        ny,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
        coarse_step=coarse_step,
        fine_step=fine_step,
        huber_delta=huber_delta,
        min_samples=min_samples,
        dso_confidence=None,
        lambda_dso=0.0,
        center_Ix=None,
        center_Iy=None,
        Ix=None,
        Iy=None,
        orientation_gamma=0.0,
        use_orientation=False,
        spatial_sigma=None
    )


def estimate_patch_normal_displacement_weighted(
    image_ref,
    image_target,
    x,
    y,
    nx,
    ny,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP,
    huber_delta=HUBER_DELTA,
    min_samples=MIN_PATCH_SAMPLES,
    dso_confidence=None,
    lambda_dso=DSO_WEIGHT_LAMBDA,
    center_Ix=None,
    center_Iy=None,
    Ix=None,
    Iy=None,
    orientation_gamma=ORIENTATION_GAMMA,
    use_orientation=False,
    spatial_sigma=None
):
    """
    Estimate scalar normal displacement with optional DSO/confidence weights.

    When lambda_dso is zero and spatial_sigma is None, this is exactly the
    ordinary patch estimator.
    """

    offsets = np.arange(
        -patch_radius,
        patch_radius + 1,
        dtype=np.float32
    )

    ox, oy = np.meshgrid(
        offsets,
        offsets
    )

    ref_x = x + ox.reshape(-1)
    ref_y = y + oy.reshape(-1)

    ref_values, ref_valid = _sample_bilinear(
        image_ref,
        ref_x,
        ref_y
    )

    if np.count_nonzero(ref_valid) < min_samples:
        return 0.0, np.inf, int(np.count_nonzero(ref_valid)), False

    ref_x = ref_x[ref_valid]
    ref_y = ref_y[ref_valid]
    ref_values = ref_values[ref_valid]

    weights = None

    if (
        dso_confidence is not None
        or spatial_sigma is not None
    ):
        weights = np.ones_like(
            ref_values,
            dtype=np.float32
        )

        if spatial_sigma is not None and spatial_sigma > 0:
            spatial_weight = np.exp(
                -(
                    (ref_x - x) ** 2
                    + (ref_y - y) ** 2
                )
                /
                (2.0 * spatial_sigma * spatial_sigma)
            )
            weights *= spatial_weight.astype(np.float32)

        if dso_confidence is not None and lambda_dso != 0:
            confidence_values, confidence_valid = _sample_bilinear(
                dso_confidence,
                ref_x,
                ref_y
            )
            confidence_values = np.where(
                confidence_valid,
                confidence_values,
                0.0
            ).astype(np.float32)

            orientation_values = np.ones_like(
                confidence_values,
                dtype=np.float32
            )

            if use_orientation:
                if Ix is None or Iy is None:
                    raise ValueError(
                        "Ix and Iy are required when use_orientation=True"
                    )

                q_ix, qx_valid = _sample_bilinear(
                    Ix,
                    ref_x,
                    ref_y
                )
                q_iy, qy_valid = _sample_bilinear(
                    Iy,
                    ref_x,
                    ref_y
                )

                q_mag = np.sqrt(q_ix**2 + q_iy**2)
                q_valid = (
                    qx_valid
                    & qy_valid
                    & (q_mag > 1e-8)
                )

                if center_Ix is None:
                    center_Ix = nx
                if center_Iy is None:
                    center_Iy = ny

                center_mag = max(
                    float(np.sqrt(center_Ix**2 + center_Iy**2)),
                    1e-8
                )
                center_nx = float(center_Ix) / center_mag
                center_ny = float(center_Iy) / center_mag

                orientation_values = np.zeros_like(
                    confidence_values,
                    dtype=np.float32
                )
                orientation_values[q_valid] = np.abs(
                    center_nx * q_ix[q_valid] / q_mag[q_valid]
                    + center_ny * q_iy[q_valid] / q_mag[q_valid]
                )

                if orientation_gamma != 1.0:
                    orientation_values = orientation_values ** orientation_gamma

            weights *= (
                1.0
                + lambda_dso * confidence_values * orientation_values
            ).astype(np.float32)

    def evaluate(displacement):
        target_x = ref_x + displacement * nx
        target_y = ref_y + displacement * ny

        target_values, target_valid = _sample_bilinear(
            image_target,
            target_x,
            target_y
        )

        num_valid = int(
            np.count_nonzero(target_valid)
        )

        if num_valid < min_samples:
            return np.inf, num_valid

        residuals = (
            target_values[target_valid]
            - ref_values[target_valid]
        )

        local_weights = (
            None
            if weights is None
            else weights[target_valid]
        )

        return _weighted_huber_mean(
            residuals,
            local_weights,
            huber_delta
        ), num_valid

    coarse_candidates = np.arange(
        -max_displacement,
        max_displacement + 0.5 * coarse_step,
        coarse_step,
        dtype=np.float32
    )

    best_displacement = 0.0
    best_error = np.inf
    best_samples = 0

    for displacement in coarse_candidates:
        error, samples = evaluate(float(displacement))

        if error < best_error:
            best_error = error
            best_displacement = float(displacement)
            best_samples = samples

    fine_start = max(
        -max_displacement,
        best_displacement - coarse_step
    )
    fine_stop = min(
        max_displacement,
        best_displacement + coarse_step
    )

    fine_candidates = np.arange(
        fine_start,
        fine_stop + 0.5 * fine_step,
        fine_step,
        dtype=np.float32
    )

    for displacement in fine_candidates:
        error, samples = evaluate(float(displacement))

        if error < best_error:
            best_error = error
            best_displacement = float(displacement)
            best_samples = samples

    success = (
        np.isfinite(best_error)
        and best_samples >= min_samples
    )

    return (
        best_displacement,
        best_error,
        best_samples,
        success
    )


# ============================================================
# Classical normal flow variants
# ============================================================

def compute_normal_flow_baseline(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next
):
    """
    Compute classical normal flow at the middle frame.

    Brightness constancy:

        Ix*u + Iy*v + It = 0

    Normal flow:

                  -It
        v_n = ------------ * grad(I)
              ||grad(I)||²


    Returns
    -------
    vx : np.ndarray
        Horizontal normal-flow component [pixels / second]

    vy : np.ndarray
        Vertical normal-flow component [pixels / second]

    magnitude : np.ndarray
        Magnitude of normal flow [pixels / second]

    valid : np.ndarray
        Boolean mask indicating pixels with sufficiently
        strong spatial gradients.

    Ix, Iy, It : np.ndarray
        Image derivatives.
    """

    Ix, Iy = compute_image_derivatives(
        I_curr
    )

    # --------------------------------------------------------
    # Temporal derivative
    # --------------------------------------------------------

    # Three-frame central difference:
    #
    #               I(t+dt) - I(t-dt)
    #       It ~= ----------------------
    #                   2 dt
    #
    # Here we use the actual timestamps.

    dt = t_next - t_prev

    if dt <= 0:
        raise ValueError(
            "Non-positive time interval between frames."
        )

    It = (I_next - I_prev) / dt

    (
        vx,
        vy,
        magnitude,
        valid
    ) = solve_normal_flow_from_derivatives(
        Ix,
        Iy,
        It
    )

    return (
        vx,
        vy,
        magnitude,
        valid,
        Ix,
        Iy,
        It
    )


def compute_normal_flow_affine(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next
):
    """
    Classical normal flow with global affine brightness compensation.

    The middle frame remains the reference. The previous and next images
    are mapped into the current frame's brightness before computing It.
    """

    Ix, Iy = compute_image_derivatives(
        I_curr
    )

    a_prev, b_prev = estimate_affine_brightness(
        I_prev,
        I_curr
    )
    a_next, b_next = estimate_affine_brightness(
        I_next,
        I_curr
    )

    I_prev_hat = apply_affine_brightness(
        I_prev,
        a_prev,
        b_prev
    )
    I_next_hat = apply_affine_brightness(
        I_next,
        a_next,
        b_next
    )

    dt = t_next - t_prev

    if dt <= 0:
        raise ValueError(
            "Non-positive time interval between frames."
        )

    It = (I_next_hat - I_prev_hat) / dt

    (
        vx,
        vy,
        magnitude,
        valid
    ) = solve_normal_flow_from_derivatives(
        Ix,
        Iy,
        It
    )

    brightness = {
        "a_prev": a_prev,
        "b_prev": b_prev,
        "a_next": a_next,
        "b_next": b_next,
    }

    return (
        vx,
        vy,
        magnitude,
        valid,
        Ix,
        Iy,
        It,
        brightness
    )


def _compute_normal_flow_patch_core(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next,
    locations=None,
    use_affine=False,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP,
    dso_confidence=None,
    lambda_dso=0.0,
    use_orientation=False,
    orientation_gamma=ORIENTATION_GAMMA,
    spatial_sigma=None
):
    """
    Patch-based direct estimation of scalar normal displacement.
    """

    Ix, Iy = compute_image_derivatives(
        I_curr
    )

    grad_sq = Ix**2 + Iy**2
    grad_mag = np.sqrt(grad_sq)
    gradient_valid = grad_mag > GRAD_THRESHOLD

    if locations is None:
        ys, xs = np.nonzero(
            gradient_valid
        )
        locations = list(
            zip(ys.tolist(), xs.tolist())
        )

    I_prev_match = I_prev
    I_next_match = I_next
    brightness = None

    if use_affine:
        a_prev, b_prev = estimate_affine_brightness(
            I_prev,
            I_curr
        )
        a_next, b_next = estimate_affine_brightness(
            I_next,
            I_curr
        )

        I_prev_match = apply_affine_brightness(
            I_prev,
            a_prev,
            b_prev
        )
        I_next_match = apply_affine_brightness(
            I_next,
            a_next,
            b_next
        )

        brightness = {
            "a_prev": a_prev,
            "b_prev": b_prev,
            "a_next": a_next,
            "b_next": b_next,
        }

    vx = np.zeros_like(I_curr)
    vy = np.zeros_like(I_curr)
    confidence = np.zeros_like(I_curr)
    photometric_error = np.full_like(
        I_curr,
        np.inf,
        dtype=np.float32
    )
    valid = np.zeros_like(
        gradient_valid,
        dtype=bool
    )

    dt_forward = t_next - t_curr
    dt_backward = t_curr - t_prev

    max_patch_samples = (
        (2 * patch_radius + 1)
        *
        (2 * patch_radius + 1)
    )

    for y, x in locations:
        if not gradient_valid[y, x]:
            continue

        inv_mag = 1.0 / max(
            float(grad_mag[y, x]),
            1e-8
        )
        nx = float(Ix[y, x] * inv_mag)
        ny = float(Iy[y, x] * inv_mag)

        scalar_values = []
        errors = []
        sample_counts = []

        if dt_forward > 0:
            (
                d_forward,
                err_forward,
                samples_forward,
                success_forward
            ) = estimate_patch_normal_displacement(
                I_curr,
                I_next_match,
                float(x),
                float(y),
                nx,
                ny,
                patch_radius=patch_radius,
                max_displacement=max_displacement,
                coarse_step=coarse_step,
                fine_step=fine_step
            ) if dso_confidence is None and spatial_sigma is None else estimate_patch_normal_displacement_weighted(
                I_curr,
                I_next_match,
                float(x),
                float(y),
                nx,
                ny,
                patch_radius=patch_radius,
                max_displacement=max_displacement,
                coarse_step=coarse_step,
                fine_step=fine_step,
                dso_confidence=dso_confidence,
                lambda_dso=lambda_dso,
                center_Ix=Ix[y, x],
                center_Iy=Iy[y, x],
                Ix=Ix,
                Iy=Iy,
                use_orientation=use_orientation,
                orientation_gamma=orientation_gamma,
                spatial_sigma=spatial_sigma
            )

            if success_forward:
                scalar_values.append(
                    d_forward / dt_forward
                )
                errors.append(
                    err_forward
                )
                sample_counts.append(
                    samples_forward
                )

        if dt_backward > 0:
            (
                d_backward,
                err_backward,
                samples_backward,
                success_backward
            ) = estimate_patch_normal_displacement(
                I_curr,
                I_prev_match,
                float(x),
                float(y),
                nx,
                ny,
                patch_radius=patch_radius,
                max_displacement=max_displacement,
                coarse_step=coarse_step,
                fine_step=fine_step
            ) if dso_confidence is None and spatial_sigma is None else estimate_patch_normal_displacement_weighted(
                I_curr,
                I_prev_match,
                float(x),
                float(y),
                nx,
                ny,
                patch_radius=patch_radius,
                max_displacement=max_displacement,
                coarse_step=coarse_step,
                fine_step=fine_step,
                dso_confidence=dso_confidence,
                lambda_dso=lambda_dso,
                center_Ix=Ix[y, x],
                center_Iy=Iy[y, x],
                Ix=Ix,
                Iy=Iy,
                use_orientation=use_orientation,
                orientation_gamma=orientation_gamma,
                spatial_sigma=spatial_sigma
            )

            if success_backward:
                scalar_values.append(
                    -d_backward / dt_backward
                )
                errors.append(
                    err_backward
                )
                sample_counts.append(
                    samples_backward
                )

        if not scalar_values:
            continue

        s_n = float(
            np.mean(scalar_values)
        )

        vx[y, x] = s_n * nx
        vy[y, x] = s_n * ny

        valid[y, x] = True

        photometric_error[y, x] = float(
            np.mean(errors)
        )

        confidence[y, x] = float(
            np.mean(sample_counts)
            /
            max_patch_samples
        )

    magnitude = np.sqrt(
        vx**2 + vy**2
    )

    if brightness is None:
        return (
            vx,
            vy,
            magnitude,
            valid,
            confidence,
            photometric_error,
            Ix,
            Iy
        )

    return (
        vx,
        vy,
        magnitude,
        valid,
        confidence,
        photometric_error,
        Ix,
        Iy,
        brightness
    )


def compute_normal_flow_patch(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next,
    locations=None,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP
):
    """
    Patch-based direct normal-displacement normal flow.
    """

    return _compute_normal_flow_patch_core(
        I_prev,
        I_curr,
        I_next,
        t_prev,
        t_curr,
        t_next,
        locations=locations,
        use_affine=False,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
        coarse_step=coarse_step,
        fine_step=fine_step
    )


def compute_normal_flow_patch_affine(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next,
    locations=None,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP
):
    """
    Patch-based direct normal displacement with affine brightness compensation.
    """

    return _compute_normal_flow_patch_core(
        I_prev,
        I_curr,
        I_next,
        t_prev,
        t_curr,
        t_next,
        locations=locations,
        use_affine=True,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
        coarse_step=coarse_step,
        fine_step=fine_step
    )


def compute_normal_flow_patch_dso(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next,
    dso_confidence,
    locations=None,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP,
    lambda_dso=DSO_WEIGHT_LAMBDA,
    spatial_sigma=None
):
    """
    Patch normal flow with DSO confidence weighting and no orientation gating.

    Setting lambda_dso=0 reduces to ordinary patch normal flow.
    """

    return _compute_normal_flow_patch_core(
        I_prev,
        I_curr,
        I_next,
        t_prev,
        t_curr,
        t_next,
        locations=locations,
        use_affine=False,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
        coarse_step=coarse_step,
        fine_step=fine_step,
        dso_confidence=dso_confidence,
        lambda_dso=lambda_dso,
        use_orientation=False,
        spatial_sigma=spatial_sigma
    )


def compute_normal_flow_patch_dso_oriented(
    I_prev,
    I_curr,
    I_next,
    t_prev,
    t_curr,
    t_next,
    dso_confidence,
    locations=None,
    patch_radius=PATCH_RADIUS,
    max_displacement=MAX_DISPLACEMENT,
    coarse_step=COARSE_STEP,
    fine_step=FINE_STEP,
    lambda_dso=DSO_WEIGHT_LAMBDA,
    orientation_gamma=ORIENTATION_GAMMA,
    spatial_sigma=None
):
    """
    Patch normal flow with DSO confidence weighting and orientation gating.
    """

    return _compute_normal_flow_patch_core(
        I_prev,
        I_curr,
        I_next,
        t_prev,
        t_curr,
        t_next,
        locations=locations,
        use_affine=False,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
        coarse_step=coarse_step,
        fine_step=fine_step,
        dso_confidence=dso_confidence,
        lambda_dso=lambda_dso,
        use_orientation=True,
        orientation_gamma=orientation_gamma,
        spatial_sigma=spatial_sigma
    )


def compute_normal_flow(*args, **kwargs):
    """
    Backward-compatible alias for the original classical estimator.
    """

    return compute_normal_flow_baseline(
        *args,
        **kwargs
    )


# ============================================================
# Arrow visualization
# ============================================================

def draw_normal_flow(
    frame,
    vx,
    vy,
    magnitude,
    valid,
    step=GRID_STEP,
    arrow_scale=ARROW_SCALE
):
    """
    Draw one normal-flow arrow per grid cell.

    Instead of blindly sampling the center of each cell,
    we select the strongest valid normal-flow measurement
    inside the cell.

    This is much better for normal flow because valid
    measurements tend to occur primarily around edges.
    """

    output = frame.copy()

    height, width = frame.shape[:2]

    max_arrow_length = (
        step * MAX_ARROW_LENGTH_FACTOR
    )

    # --------------------------------------------------------
    # Visit each grid cell
    # --------------------------------------------------------

    for y0 in range(0, height, step):

        for x0 in range(0, width, step):

            y1 = min(
                y0 + step,
                height
            )

            x1 = min(
                x0 + step,
                width
            )

            # Extract valid mask for this grid cell
            cell_valid = valid[
                y0:y1,
                x0:x1
            ]

            if not np.any(cell_valid):
                continue

            # ------------------------------------------------
            # Find strongest valid normal-flow point
            # ------------------------------------------------

            cell_magnitude = magnitude[
                y0:y1,
                x0:x1
            ].copy()

            # Invalid pixels cannot be selected
            cell_magnitude[
                ~cell_valid
            ] = -1.0

            relative_y, relative_x = np.unravel_index(
                np.argmax(cell_magnitude),
                cell_magnitude.shape
            )

            y = y0 + relative_y
            x = x0 + relative_x

            fx = vx[y, x]
            fy = vy[y, x]

            if (
                not np.isfinite(fx)
                or not np.isfinite(fy)
            ):
                continue

            # ------------------------------------------------
            # Scale arrow for visualization
            # ------------------------------------------------

            dx = arrow_scale * fx
            dy = arrow_scale * fy

            arrow_length = np.hypot(
                dx,
                dy
            )

            # Clip very long arrows
            if arrow_length > max_arrow_length:

                scale = (
                    max_arrow_length
                    /
                    arrow_length
                )

                dx *= scale
                dy *= scale

            # Ignore arrows so short that they are meaningless
            if np.hypot(dx, dy) < 1.0:
                continue

            # ------------------------------------------------
            # Draw arrow
            # ------------------------------------------------

            start_point = (
                int(x),
                int(y)
            )

            end_point = (
                int(round(x + dx)),
                int(round(y + dy))
            )

            cv2.arrowedLine(
                output,
                start_point,
                end_point,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
                tipLength=0.3
            )

            # Mark the origin of the normal-flow measurement
            cv2.circle(
                output,
                start_point,
                2,
                (0, 0, 255),
                -1
            )

    return output


# ============================================================
# Main camera loop
# ============================================================

def main():

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"{CAMERA_INDEX}"
        )

    # Camera resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    # --------------------------------------------------------
    # Read first frame
    # --------------------------------------------------------

    ret, frame_prev = cap.read()

    if not ret:

        cap.release()

        raise RuntimeError(
            "Could not read first camera frame."
        )

    t_prev = time.perf_counter()

    I_prev = preprocess(
        frame_prev
    )

    # --------------------------------------------------------
    # Read second frame
    # --------------------------------------------------------

    ret, frame_curr = cap.read()

    if not ret:

        cap.release()

        raise RuntimeError(
            "Could not read second camera frame."
        )

    t_curr = time.perf_counter()

    I_curr = preprocess(
        frame_curr
    )

    # --------------------------------------------------------
    # FPS tracking
    # --------------------------------------------------------

    fps_filtered = 0.0
    method_index = 0
    method_names = [
        "Baseline",
        "Affine",
        "Patch",
        "Patch + Affine"
    ]

    print(
        "Normal flow camera started."
    )

    print(
        "Press 'm' to cycle methods. Press 'q' or ESC to quit."
    )

    # ========================================================
    # Main streaming loop
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Get next frame
        # ----------------------------------------------------

        ret, frame_next = cap.read()

        if not ret:

            print(
                "Could not read camera frame."
            )

            break

        t_next = time.perf_counter()

        I_next = preprocess(
            frame_next
        )

        # ----------------------------------------------------
        # Compute normal flow
        # ----------------------------------------------------

        method_name = method_names[
            method_index
        ]

        brightness = None
        photometric_error = None

        if method_name == "Baseline":
            (
                vx,
                vy,
                magnitude,
                valid,
                Ix,
                Iy,
                It
            ) = compute_normal_flow_baseline(

                I_prev,
                I_curr,
                I_next,

                t_prev,
                t_curr,
                t_next
            )

        elif method_name == "Affine":
            (
                vx,
                vy,
                magnitude,
                valid,
                Ix,
                Iy,
                It,
                brightness
            ) = compute_normal_flow_affine(

                I_prev,
                I_curr,
                I_next,

                t_prev,
                t_curr,
                t_next
            )

        else:
            Ix, Iy = compute_image_derivatives(
                I_curr
            )
            grad_mag = np.sqrt(
                Ix**2 + Iy**2
            )
            selected_valid = (
                grad_mag > GRAD_THRESHOLD
            )
            locations = select_strongest_gradient_points(
                selected_valid,
                grad_mag,
                step=GRID_STEP
            )

            if method_name == "Patch":
                (
                    vx,
                    vy,
                    magnitude,
                    valid,
                    confidence,
                    photometric_error,
                    Ix,
                    Iy
                ) = compute_normal_flow_patch(

                    I_prev,
                    I_curr,
                    I_next,

                    t_prev,
                    t_curr,
                    t_next,
                    locations=locations
                )

            else:
                (
                    vx,
                    vy,
                    magnitude,
                    valid,
                    confidence,
                    photometric_error,
                    Ix,
                    Iy,
                    brightness
                ) = compute_normal_flow_patch_affine(

                    I_prev,
                    I_curr,
                    I_next,

                    t_prev,
                    t_curr,
                    t_next,
                    locations=locations
                )

        # ----------------------------------------------------
        # Draw normal-flow arrows
        # ----------------------------------------------------

        flow_display = draw_normal_flow(

            frame_curr,

            vx,
            vy,
            magnitude,
            valid
        )

        # ----------------------------------------------------
        # FPS calculation
        # ----------------------------------------------------

        frame_dt = (
            t_next - t_curr
        )

        if frame_dt > 0:

            fps = 1.0 / frame_dt

            if fps_filtered == 0:

                fps_filtered = fps

            else:

                fps_filtered = (
                    0.9 * fps_filtered
                    +
                    0.1 * fps
                )

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        valid_count = np.count_nonzero(
            valid
        )

        valid_fraction = (
            valid_count
            /
            valid.size
        )

        mean_magnitude = (
            float(np.mean(magnitude[valid]))
            if valid_count > 0
            else 0.0
        )

        cv2.putText(
            flow_display,
            f"FPS: {fps_filtered:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            flow_display,
            f"Valid: {valid_count} ({100 * valid_fraction:.1f}%)",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            flow_display,
            f"{method_name} | Mean |v_n|: {mean_magnitude:.2f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if photometric_error is not None:
            photo_valid = (
                valid
                & np.isfinite(photometric_error)
            )
            mean_photo = (
                float(np.mean(photometric_error[photo_valid]))
                if np.any(photo_valid)
                else 0.0
            )

            cv2.putText(
                flow_display,
                f"Mean photo residual: {mean_photo:.5f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        elif brightness is not None:
            cv2.putText(
                flow_display,
                (
                    f"a_next: {brightness['a_next']:.3f} "
                    f"b_next: {brightness['b_next']:.3f}"
                ),
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Show live camera
        # ----------------------------------------------------

        cv2.imshow(
            "RGB Normal Flow",
            flow_display
        )

        # ----------------------------------------------------
        # Shift three-frame buffer
        #
        # prev <- curr
        # curr <- next
        # ----------------------------------------------------

        frame_prev = frame_curr
        frame_curr = frame_next

        I_prev = I_curr
        I_curr = I_next

        t_prev = t_curr
        t_curr = t_next

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if (
            key == ord("q")
            or key == 27
        ):
            break

        if key == ord("m"):
            method_index = (
                method_index + 1
            ) % len(method_names)

    # ========================================================
    # Cleanup
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
