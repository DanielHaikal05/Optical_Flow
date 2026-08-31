#!/usr/bin/env python3
"""
Generate ground-truth normal flow for a TartanAir V1 trajectory.

Expected trajectory structure:
trajectory/
├── image_left/
│   ├── 000000_left.png
│   ├── 000001_left.png
│   └── ...
└── flow/
    ├── 000000_000001_flow.npy
    ├── 000000_000001_mask.npy
    └── ...

For every optical-flow file, the script computes

    n = ((u · grad(I)) / ||grad(I)||^2) grad(I)

where:
    u       is the GT optical-flow vector [u, v],
    grad(I) is the spatial gradient of the first RGB frame,
    n       is the GT normal-flow vector [n_x, n_y].

Outputs:
output_dir/
├── vector/        H x W x 2 float32 normal-flow vectors
├── scalar/        H x W float32 signed normal-flow components
├── gradient_dir/  H x W x 2 float32 unit image-gradient directions
├── valid_mask/    H x W uint8 masks, 255 = valid
└── preview/       optional color visualizations

Dependencies:
    pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


FLOW_RE = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)_flow\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GT normal flow from TartanAir RGB and GT optical flow."
    )
    parser.add_argument(
        "trajectory_root",
        type=Path,
        help="Trajectory folder containing image_left/ and flow/.",
    )
    parser.add_argument(
        "--image-dir",
        default="image_left",
        help="RGB-image folder relative to trajectory_root (default: image_left).",
    )
    parser.add_argument(
        "--flow-dir",
        default="flow",
        help="Optical-flow folder relative to trajectory_root (default: flow).",
    )
    parser.add_argument(
        "--output-dir",
        default="normal_flow_gt",
        help="Output folder relative to trajectory_root, or an absolute path.",
    )
    parser.add_argument(
        "--min-gradient",
        type=float,
        default=0.01,
        help=(
            "Minimum gradient magnitude for a valid normal-flow label. "
            "Images are scaled to [0,1] before Sobel filtering (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Numerical epsilon used when normalizing gradients.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("zero-invalid", "zero-valid", "ignore"),
        default="zero-valid",
        help=(
            "Interpretation of TartanAir *_mask.npy files. "
            "'zero-invalid': mask != 0 is valid; "
            "'zero-valid': mask == 0 is valid; "
            "'ignore': do not use the supplied mask. "
            "Default: zero-invalid."
        ),
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian blur sigma before computing gradients (default: 0).",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save an HSV color visualization for each normal-flow field.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output arrays.",
    )
    return parser.parse_args()


def resolve_output_dir(root: Path, output_arg: str) -> Path:
    output = Path(output_arg)
    return output if output.is_absolute() else root / output


def find_first_image(image_dir: Path, frame_id: str) -> Path:
    candidates = [
        image_dir / f"{frame_id}_left.png",
        image_dir / f"{frame_id}.png",
        image_dir / f"{frame_id}_left.jpg",
        image_dir / f"{frame_id}.jpg",
        image_dir / f"{frame_id}_left.jpeg",
        image_dir / f"{frame_id}.jpeg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No RGB image found for frame {frame_id}. Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def load_rgb_grayscale(image_path: Path, blur_sigma: float) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if blur_sigma > 0:
        gray = cv2.GaussianBlur(
            gray,
            ksize=(0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )

    return gray


def compute_image_gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Spatial derivatives in pixel coordinates:
    # +x points right and +y points down, matching optical-flow convention.
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=3,
        scale=1.0 / 8.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=3,
        scale=1.0 / 8.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    return gx, gy


def normalize_flow_shape(flow: np.ndarray, path: Path) -> np.ndarray:
    flow = np.asarray(flow)

    if flow.ndim != 3:
        raise ValueError(
            f"Expected a 3D flow array in {path}, but got shape {flow.shape}."
        )

    if flow.shape[-1] >= 2:
        flow = flow[..., :2]
    elif flow.shape[0] >= 2:
        flow = np.moveaxis(flow[:2], 0, -1)
    else:
        raise ValueError(
            f"Could not interpret flow channels in {path}; shape is {flow.shape}."
        )

    return flow.astype(np.float32, copy=False)


def load_tartanair_mask(
    mask_path: Path,
    expected_shape: tuple[int, int],
    mode: str,
) -> np.ndarray:
    if mode == "ignore" or not mask_path.is_file():
        return np.ones(expected_shape, dtype=bool)

    mask = np.asarray(np.load(mask_path))

    # Handle HxWx1 masks.
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match flow/image shape "
            f"{expected_shape}: {mask_path}"
        )

    if mode == "zero-invalid":
        return mask != 0
    if mode == "zero-valid":
        return mask == 0

    raise ValueError(f"Unsupported mask mode: {mode}")


def compute_normal_flow(
    flow: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    min_gradient: float,
    epsilon: float,
    supplied_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        normal_vector: H x W x 2
        normal_scalar: H x W signed projection u dot g
        gradient_dir:  H x W x 2 unit gradient direction
        valid:         H x W bool
    """
    if flow.shape[:2] != gx.shape or gx.shape != gy.shape:
        raise ValueError(
            "Image and flow resolutions differ: "
            f"flow={flow.shape[:2]}, gx={gx.shape}, gy={gy.shape}"
        )

    grad_magnitude = np.sqrt(gx * gx + gy * gy)
    strong_gradient = grad_magnitude >= min_gradient

    finite = (
        np.isfinite(flow[..., 0])
        & np.isfinite(flow[..., 1])
        & np.isfinite(gx)
        & np.isfinite(gy)
    )

    valid = strong_gradient & finite & supplied_valid_mask

    safe_magnitude = np.maximum(grad_magnitude, epsilon)
    gradient_dir = np.stack(
        (gx / safe_magnitude, gy / safe_magnitude),
        axis=-1,
    ).astype(np.float32)

    # Signed normal-flow scalar:
    # s = optical_flow dot unit_gradient.
    normal_scalar = (
        flow[..., 0] * gradient_dir[..., 0]
        + flow[..., 1] * gradient_dir[..., 1]
    ).astype(np.float32)

    # Reconstruct the 2D vector representation n = s g.
    normal_vector = (
        normal_scalar[..., None] * gradient_dir
    ).astype(np.float32)

    # Invalid/undefined pixels are zeroed and identified by valid_mask.
    normal_scalar[~valid] = 0.0
    normal_vector[~valid] = 0.0
    gradient_dir[~valid] = 0.0

    return normal_vector, normal_scalar, gradient_dir, valid


def normal_flow_to_bgr(
    normal_flow: np.ndarray,
    valid: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    nx = normal_flow[..., 0]
    ny = normal_flow[..., 1]

    magnitude, angle = cv2.cartToPolar(nx, ny, angleInDegrees=True)

    valid_magnitudes = magnitude[valid]
    scale = (
        float(np.percentile(valid_magnitudes, percentile))
        if valid_magnitudes.size
        else 1.0
    )
    scale = max(scale, 1e-6)

    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
    hsv[..., 2] = np.clip(magnitude / scale * 255.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def output_paths(output_root: Path, pair_stem: str) -> dict[str, Path]:
    paths = {
        "vector": output_root / "vector" / f"{pair_stem}_normal_flow.npy",
        "scalar": output_root / "scalar" / f"{pair_stem}_normal_scalar.npy",
        "gradient_dir": (
            output_root / "gradient_dir" / f"{pair_stem}_gradient_dir.npy"
        ),
        "valid_mask": output_root / "valid_mask" / f"{pair_stem}_valid.png",
        "preview": output_root / "preview" / f"{pair_stem}_normal_flow.png",
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    return paths


def process_trajectory(args: argparse.Namespace) -> None:
    root = args.trajectory_root.expanduser().resolve()
    image_dir = root / args.image_dir
    flow_dir = root / args.flow_dir
    output_root = resolve_output_dir(root, args.output_dir).expanduser().resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not flow_dir.is_dir():
        raise FileNotFoundError(f"Flow directory does not exist: {flow_dir}")

    flow_files = sorted(flow_dir.glob("*_flow.npy"))
    if not flow_files:
        raise FileNotFoundError(f"No *_flow.npy files found in {flow_dir}")

    completed = 0
    skipped = 0

    for index, flow_path in enumerate(flow_files, start=1):
        match = FLOW_RE.match(flow_path.name)
        if match is None:
            print(f"[skip] Unexpected flow filename: {flow_path.name}")
            skipped += 1
            continue

        first_id = match.group("first")
        second_id = match.group("second")
        pair_stem = f"{first_id}_{second_id}"

        paths = output_paths(output_root, pair_stem)
        required_outputs = [
            paths["vector"],
            paths["scalar"],
            paths["gradient_dir"],
            paths["valid_mask"],
        ]

        if (
            not args.overwrite
            and all(path.exists() for path in required_outputs)
            and (not args.save_preview or paths["preview"].exists())
        ):
            print(f"[{index}/{len(flow_files)}] exists: {pair_stem}")
            skipped += 1
            continue

        image_path = find_first_image(image_dir, first_id)
        mask_path = flow_dir / f"{pair_stem}_mask.npy"

        flow = normalize_flow_shape(np.load(flow_path), flow_path)
        gray = load_rgb_grayscale(image_path, args.blur_sigma)

        if gray.shape != flow.shape[:2]:
            raise ValueError(
                f"Resolution mismatch for {pair_stem}: "
                f"image={gray.shape}, flow={flow.shape[:2]}"
            )

        gx, gy = compute_image_gradient(gray)
        supplied_mask = load_tartanair_mask(
            mask_path=mask_path,
            expected_shape=gray.shape,
            mode=args.mask_mode,
        )

        normal_vector, normal_scalar, gradient_dir, valid = compute_normal_flow(
            flow=flow,
            gx=gx,
            gy=gy,
            min_gradient=args.min_gradient,
            epsilon=args.epsilon,
            supplied_valid_mask=supplied_mask,
        )

        np.save(paths["vector"], normal_vector)
        np.save(paths["scalar"], normal_scalar)
        np.save(paths["gradient_dir"], gradient_dir)

        valid_u8 = valid.astype(np.uint8) * 255
        if not cv2.imwrite(str(paths["valid_mask"]), valid_u8):
            raise IOError(f"Could not save mask: {paths['valid_mask']}")

        if args.save_preview:
            preview = normal_flow_to_bgr(normal_vector, valid)
            if not cv2.imwrite(str(paths["preview"]), preview):
                raise IOError(f"Could not save preview: {paths['preview']}")

        valid_fraction = 100.0 * float(valid.mean())
        print(
            f"[{index}/{len(flow_files)}] {pair_stem}: "
            f"saved, {valid_fraction:.1f}% valid"
        )
        completed += 1

    print(
        f"\nDone. Generated {completed} pairs, skipped {skipped}. "
        f"Output: {output_root}"
    )


def main() -> int:
    args = parse_args()

    if args.min_gradient < 0:
        print("--min-gradient must be nonnegative.", file=sys.stderr)
        return 2
    if args.epsilon <= 0:
        print("--epsilon must be positive.", file=sys.stderr)
        return 2
    if args.blur_sigma < 0:
        print("--blur-sigma must be nonnegative.", file=sys.stderr)
        return 2

    try:
        process_trajectory(args)
    except (FileNotFoundError, ValueError, IOError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
