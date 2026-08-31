#!/usr/bin/env python3
"""Run the EV-IMO DVS implementation on DSEC event streams.

This wraps yechengxi/DVS directly. It renders every DSEC event into fixed-rate
event frames, runs ECN_Disp and ECN_Pose, and writes depth, ego-flow, pose, and
explainability/motion masks. No event subsampling is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DSEC_ROOT = ROOT / "Datasets" / "DSEC"
DVS_ROOT = ROOT / "DVS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsec-root", type=Path, default=DSEC_ROOT)
    parser.add_argument("--dvs-root", type=Path, default=DVS_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "DVS" / "outputs" / "dsec_evimo")
    parser.add_argument("--dispnet", type=Path, required=True)
    parser.add_argument("--posenet", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["zurich_city_00_a", "interlaken_00_c", "zurich_city_00_b"],
    )
    parser.add_argument("--interval-ms", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--img-height", type=int, default=200)
    parser.add_argument("--img-width", type=int, default=346)
    parser.add_argument("--arch", choices=["ecn", "std"], default="ecn")
    parser.add_argument("--norm-type", default="fd")
    parser.add_argument("--norm-group", type=int, default=16)
    parser.add_argument("--n-channel", type=int, default=8)
    parser.add_argument("--growth-rate", type=int, default=8)
    parser.add_argument("--scale-factor", type=float, default=0.5)
    parser.add_argument("--final-map-size", type=int, default=4)
    parser.add_argument("--preview-stride", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_dvs_to_path(dvs_root: Path) -> None:
    if not dvs_root.exists():
        raise FileNotFoundError(f"DVS repo not found: {dvs_root}")
    sys.path.insert(0, str(dvs_root))


def strip_module_prefix(state_dict: dict) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def load_state(path: Path) -> OrderedDict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(str(path), map_location="cpu")
    state = obj.get("state_dict") if isinstance(obj, dict) and "state_dict" in obj else obj
    if not hasattr(state, "keys"):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    return strip_module_prefix(state)


def find_events_file(seq_dir: Path) -> Path:
    preferred = [
        seq_dir / "events.h5",
        seq_dir / "event_data" / "left" / "events.h5",
        seq_dir / "Events" / "events.h5",
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(seq_dir.glob("**/events.h5"))
    if not matches:
        raise FileNotFoundError(f"No events.h5 below {seq_dir}")
    return matches[0]


def find_calibration_file(seq_dir: Path) -> Path:
    preferred = [
        seq_dir / "cam_to_cam.yaml",
        seq_dir / "calibration" / "cam_to_cam.yaml",
        seq_dir / "Calibration" / "cam_to_cam.yaml",
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(seq_dir.glob("**/cam_to_cam.yaml"))
    if not matches:
        raise FileNotFoundError(f"No cam_to_cam.yaml below {seq_dir}")
    return matches[0]


def load_scaled_intrinsics(calib_path: Path, img_width: int, img_height: int) -> np.ndarray:
    data = yaml.safe_load(calib_path.read_text())
    cam = data["intrinsics"]["camRect0"]
    fx, fy, cx, cy = [float(v) for v in cam["camera_matrix"]]
    src_w, src_h = [float(v) for v in cam["resolution"]]
    sx = img_width / src_w
    sy = img_height / src_h
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def frame_count(ms_to_idx: np.ndarray, interval_ms: int) -> int:
    return max(0, (len(ms_to_idx) - 1) // interval_ms)


def render_event_frame(
    h5: h5py.File,
    ms_to_idx: np.ndarray,
    frame_idx: int,
    interval_ms: int,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    src_h, src_w = 480, 640
    start_ms = frame_idx * interval_ms
    end_ms = min((frame_idx + 1) * interval_ms, len(ms_to_idx) - 1)
    start = int(ms_to_idx[start_ms])
    end = int(ms_to_idx[end_ms])
    image = np.zeros((img_height, img_width, 3), dtype=np.float32)
    if end <= start:
        return image.astype(np.uint8)

    xs = h5["events/x"][start:end].astype(np.int64)
    ys = h5["events/y"][start:end].astype(np.int64)
    ps = h5["events/p"][start:end].astype(np.uint8)
    ts = h5["events/t"][start:end].astype(np.float32)

    xo = np.clip((xs * img_width) // src_w, 0, img_width - 1)
    yo = np.clip((ys * img_height) // src_h, 0, img_height - 1)
    pos = ps > 0
    neg = ~pos

    pos_counts = np.zeros((img_height, img_width), dtype=np.float32)
    neg_counts = np.zeros((img_height, img_width), dtype=np.float32)
    np.add.at(pos_counts, (yo[pos], xo[pos]), 1.0)
    np.add.at(neg_counts, (yo[neg], xo[neg]), 1.0)

    last_t = np.zeros((img_height, img_width), dtype=np.float32)
    if len(ts):
        rel_t = ts - ts.min()
        denom = max(float(rel_t.max()), 1.0)
        np.maximum.at(last_t, (yo, xo), rel_t / denom)

    for channel, counts in [(0, pos_counts), (2, neg_counts)]:
        if counts.max() > 0:
            image[:, :, channel] = np.log1p(counts) / np.log1p(counts.max())
    image[:, :, 1] = last_t
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def tensor_from_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def depth_to_png(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        vals = depth[finite]
        lo, hi = np.percentile(vals, [1, 99])
        if hi > lo:
            out[finite] = np.clip((depth[finite] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hue = (ang + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)
    hsv = np.stack([hue, sat, val], axis=-1)
    import colorsys

    flat = hsv.reshape(-1, 3)
    rgb = np.array([colorsys.hsv_to_rgb(*x) for x in flat], dtype=np.float32)
    return np.clip(rgb.reshape(hsv.shape) * 255.0, 0, 255).astype(np.uint8)


def build_models(args: argparse.Namespace, device: torch.device):
    add_dvs_to_path(args.dvs_root)
    from models.DispNetS import DispNetS
    from models.ECN import ECN_Disp, ECN_Pose
    from models.PoseExpNet import PoseExpNet

    if args.arch == "ecn":
        disp_net = ECN_Disp(
            input_size=args.img_height,
            init_planes=args.n_channel,
            scale_factor=args.scale_factor,
            growth_rate=args.growth_rate,
            final_map_size=args.final_map_size,
            norm_type=args.norm_type,
            norm_group=args.norm_group,
        ).to(device)
        pose_net = ECN_Pose(
            input_size=args.img_height,
            nb_ref_imgs=args.sequence_length - 1,
            init_planes=args.n_channel // 2,
            scale_factor=args.scale_factor,
            growth_rate=args.growth_rate // 2,
            final_map_size=args.final_map_size,
            output_exp=True,
            norm_type=args.norm_type,
            norm_group=args.norm_group,
        ).to(device)
    else:
        disp_net = DispNetS().to(device)
        pose_net = PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=True).to(device)

    disp_net.load_state_dict(load_state(args.dispnet), strict=True)
    pose_net.load_state_dict(load_state(args.posenet), strict=True)
    disp_net.eval()
    pose_net.eval()
    return disp_net, pose_net


def run_sequence(args: argparse.Namespace, seq_name: str, disp_net, pose_net, device: torch.device) -> dict:
    from inverse_warp import get_new_grid

    seq_dir = args.dsec_root / seq_name
    event_path = find_events_file(seq_dir)
    calib_path = find_calibration_file(seq_dir)
    out_dir = args.output_root / seq_name
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_np = load_scaled_intrinsics(calib_path, args.img_width, args.img_height)
    np.savetxt(out_dir / "cam.txt", intrinsics_np, fmt="%.9f")
    intrinsics = torch.from_numpy(intrinsics_np).unsqueeze(0).to(device)
    intrinsics_inv = torch.inverse(intrinsics)

    demi = (args.sequence_length - 1) // 2
    shifts = list(range(-demi, demi + 1))
    shifts.pop(demi)

    written = 0
    with h5py.File(event_path, "r") as h5:
        ms_to_idx = h5["ms_to_idx"][:]
        n_frames = frame_count(ms_to_idx, args.interval_ms)
        cache: dict[int, np.ndarray] = {}

        for i in range(demi, n_frames - demi):
            needed = [i] + [i + s for s in shifts]
            for idx in needed:
                if idx not in cache:
                    cache[idx] = render_event_frame(
                        h5, ms_to_idx, idx, args.interval_ms, args.img_width, args.img_height
                    )
            for idx in list(cache):
                if idx < i - demi:
                    del cache[idx]

            frame = cache[i]
            ref_frames = [cache[i + s] for s in shifts]
            img = tensor_from_frame(frame, device)
            refs = [tensor_from_frame(ref, device) for ref in ref_frames]

            with torch.no_grad():
                disp = disp_net(img)
                mean_disp = disp.view(1, -1).mean(-1).view(1, 1, 1, 1) * 0.1
                disp = disp / mean_disp.clamp_min(1e-6)
                depth = 1.0 / disp
                exp_mask, pose = pose_net(img, refs)
                _, ego_flow = get_new_grid(
                    depth[:, 0],
                    pose[:, int((args.sequence_length - 1) / 2)],
                    intrinsics,
                    intrinsics_inv,
                )

            stem = f"{i:06d}"
            depth_np = depth[0, 0].detach().cpu().numpy()
            exp_np = exp_mask[0].detach().cpu().numpy()
            motion_np = 1.0 - exp_np.mean(axis=0)
            flow_np = ego_flow[0].detach().cpu().numpy()

            save_npy(out_dir / "depth" / f"{stem}.npy", depth_np)
            save_npy(out_dir / "explainability" / f"{stem}.npy", exp_np)
            save_npy(out_dir / "motion_probability" / f"{stem}.npy", motion_np)
            save_npy(out_dir / "ego_flow" / f"{stem}.npy", flow_np)
            save_npy(out_dir / "pose" / f"{stem}.npy", pose[0].detach().cpu().numpy())

            save_png(out_dir / "motion_probability_png" / f"{stem}.png", (motion_np * 255).astype(np.uint8))
            if args.preview_stride > 0 and (written % args.preview_stride == 0):
                save_png(out_dir / "input_preview" / f"{stem}.jpg", frame)
                save_png(out_dir / "depth_png" / f"{stem}.png", depth_to_png(depth_np))
                save_png(out_dir / "ego_flow_png" / f"{stem}.png", flow_to_rgb(flow_np))
            written += 1

            if written % 100 == 0:
                print(f"[{seq_name}] wrote {written}/{max(n_frames - 2 * demi, 0)}", flush=True)

    summary = {
        "sequence": seq_name,
        "sequence_dir": str(seq_dir),
        "events_file": str(event_path),
        "calibration_file": str(calib_path),
        "output_dir": str(out_dir),
        "interval_ms": args.interval_ms,
        "sequence_length": args.sequence_length,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "frames_written": written,
        "dispnet": str(args.dispnet),
        "posenet": str(args.posenet),
        "outputs": [
            "depth/*.npy",
            "explainability/*.npy",
            "motion_probability/*.npy",
            "motion_probability_png/*.png",
            "ego_flow/*.npy",
            "pose/*.npy",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.sequence_length % 2 != 1 or args.sequence_length < 3:
        raise ValueError("--sequence-length must be an odd integer >= 3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disp_net, pose_net = build_models(args, device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [run_sequence(args, seq, disp_net, pose_net, device) for seq in args.sequences]
    (args.output_root / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
