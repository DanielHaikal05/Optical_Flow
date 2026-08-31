#!/usr/bin/env python3
"""Convert Prophesee EVT3 .raw event-camera recordings into PNG frames.

The recordings in ``Datasets/Recordings_from_matrice`` are not raster image
frames. Their headers identify them as Prophesee EVT3 streams from an IMX636
event camera. This script converts those asynchronous events into time-binned
PNG frames and prints useful statistics.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


DEFAULT_DATASET_DIR = Path("Datasets/Recordings_from_matrice")
DEFAULT_OUTPUT_DIR = Path("Datasets/Drone_footage/4")
DEFAULT_FRAME_MS = 33.333


EVT_TYPES = {
    0x0: "EVT_ADDR_Y",
    0x2: "EVT_ADDR_X",
    0x3: "VECT_BASE_X",
    0x4: "VECT_12",
    0x5: "VECT_8",
    0x6: "EVT_TIME_LOW",
    0x7: "CONTINUED_4",
    0x8: "EVT_TIME_HIGH",
    0xA: "EXT_TRIGGER",
    0xE: "OTHERS",
    0xF: "CONTINUED_12",
}


@dataclass
class RawHeader:
    path: Path
    offset: int
    lines: list[str]
    metadata: dict[str, str]
    width: int
    height: int
    format_name: str


@dataclass
class DecodeStats:
    words: int = 0
    ignored_trailing_bytes: int = 0
    type_counts: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.int64))
    decoded_events: int = 0
    kept_events: int = 0
    first_ts_us: int | None = None
    last_ts_us: int | None = None
    kept_first_rel_us: int | None = None
    kept_last_rel_us: int | None = None
    on_events: int = 0
    off_events: int = 0
    min_x: int | None = None
    max_x: int | None = None
    min_y: int | None = None
    max_y: int | None = None
    out_of_bounds_events: int = 0
    time_high_wraps: int = 0
    stop_reason: str = "eof"

    def update_events(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ps: np.ndarray,
        ts: np.ndarray,
        rel_ts: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        if xs.size == 0:
            return

        self.kept_events += int(xs.size)
        self.kept_first_rel_us = int(rel_ts[0]) if self.kept_first_rel_us is None else self.kept_first_rel_us
        self.kept_last_rel_us = int(rel_ts[-1])
        self.on_events += int(np.count_nonzero(ps))
        self.off_events += int(ps.size - np.count_nonzero(ps))

        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        self.out_of_bounds_events += int(valid.size - np.count_nonzero(valid))
        if np.any(valid):
            vx = xs[valid]
            vy = ys[valid]
            self.min_x = int(vx.min()) if self.min_x is None else min(self.min_x, int(vx.min()))
            self.max_x = int(vx.max()) if self.max_x is None else max(self.max_x, int(vx.max()))
            self.min_y = int(vy.min()) if self.min_y is None else min(self.min_y, int(vy.min()))
            self.max_y = int(vy.max()) if self.max_y is None else max(self.max_y, int(vy.max()))


class ProgressBar:
    def __init__(self, label: str, total: int, *, width: int = 32, min_interval_s: float = 0.1) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.width = width
        self.min_interval_s = min_interval_s
        self.last_print_s = 0.0
        self.last_line_len = 0
        self.completed = 0

    def update(self, completed: int, *, force: bool = False) -> None:
        now = time.monotonic()
        completed = min(max(int(completed), 0), self.total)
        self.completed = completed
        if not force and completed < self.total and now - self.last_print_s < self.min_interval_s:
            return

        ratio = completed / self.total
        filled = min(int(round(self.width * ratio)), self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r{self.label} [{bar}] {completed}/{self.total} frames ({ratio * 100:5.1f}%)"
        padding = " " * max(self.last_line_len - len(line), 0)
        sys.stdout.write(line + padding)
        sys.stdout.flush()
        self.last_print_s = now
        self.last_line_len = len(line)

    def finish(self) -> None:
        if self.completed != self.total:
            self.update(self.total, force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


@dataclass
class EventFrameWriter:
    width: int
    height: int
    frame_us: int
    output_dir: Path
    total_frames: int
    progress: ProgressBar

    current_frame: int = 0
    pos_image: np.ndarray = field(init=False)
    neg_image: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.pos_image = np.zeros((self.height, self.width), dtype=np.uint32)
        self.neg_image = np.zeros((self.height, self.width), dtype=np.uint32)
        self.progress.update(0, force=True)

    def add(self, xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, rel_ts: np.ndarray) -> None:
        if xs.size == 0:
            return

        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs = xs[valid]
        ys = ys[valid]
        ps = ps[valid]
        rel_ts = rel_ts[valid]
        if xs.size == 0:
            return

        frame_ids = rel_ts // self.frame_us
        change_points = np.flatnonzero(np.diff(frame_ids)) + 1
        starts = np.concatenate(([0], change_points))
        stops = np.concatenate((change_points, [frame_ids.size]))

        for start, stop in zip(starts, stops):
            frame_id = int(frame_ids[start])
            if frame_id >= self.total_frames:
                continue
            self._advance_to(frame_id)
            self._accumulate(xs[start:stop], ys[start:stop], ps[start:stop])

    def finalize(self) -> None:
        self._advance_to(self.total_frames)
        self.progress.finish()

    def _advance_to(self, frame_id: int) -> None:
        while self.current_frame < min(frame_id, self.total_frames):
            frame_path = self.output_dir / f"frame_{self.current_frame:06d}.png"
            save_event_image(self.pos_image, self.neg_image, frame_path)
            self.pos_image.fill(0)
            self.neg_image.fill(0)
            self.current_frame += 1
            self.progress.update(self.current_frame)

    def _accumulate(self, xs: np.ndarray, ys: np.ndarray, ps: np.ndarray) -> None:
        on_mask = ps.astype(bool)
        np.add.at(self.pos_image, (ys[on_mask], xs[on_mask]), 1)
        np.add.at(self.neg_image, (ys[~on_mask], xs[~on_mask]), 1)


class StopDecoding(Exception):
    pass


def parse_header(path: Path) -> RawHeader:
    lines: list[str] = []
    offset = 0

    with path.open("rb") as raw:
        while True:
            line = raw.readline()
            if not line:
                raise ValueError(f"{path} ended before a '% end' header marker was found")
            offset += len(line)
            if not line.startswith(b"%"):
                raise ValueError(f"{path} does not look like a Prophesee RAW file")
            text = line.decode("ascii", errors="replace").strip()
            lines.append(text)
            if text == "% end":
                break

    metadata: dict[str, str] = {}
    for line in lines:
        if line == "% end":
            continue
        body = line[1:].strip()
        key, _, value = body.partition(" ")
        metadata[key] = value.strip()

    width, height = parse_geometry(metadata)
    format_name = metadata.get("format", metadata.get("evt", "unknown"))
    return RawHeader(path, offset, lines, metadata, width, height, format_name)


def parse_geometry(metadata: dict[str, str]) -> tuple[int, int]:
    for value in (metadata.get("format", ""), metadata.get("geometry", "")):
        parts = value.replace(";", " ").replace("x", " ").split()
        width = height = None
        for part in parts:
            if part.startswith("width="):
                width = int(part.split("=", 1)[1])
            elif part.startswith("height="):
                height = int(part.split("=", 1)[1])
        if width and height:
            return width, height

        numeric = [int(part) for part in parts if part.isdigit()]
        if len(numeric) >= 2:
            return numeric[0], numeric[1]

    raise ValueError("Could not infer sensor geometry from RAW header")


def scan_evt3_time_bounds(header: RawHeader, *, chunk_bytes: int) -> DecodeStats:
    stats = DecodeStats()
    time_high = 0
    last_time_high_12 = None
    current_ts = 0
    carry = b""

    def note_events(count: int = 1) -> None:
        if count <= 0:
            return
        stats.decoded_events += count
        stats.first_ts_us = current_ts if stats.first_ts_us is None else stats.first_ts_us
        stats.last_ts_us = current_ts

    with header.path.open("rb") as raw:
        raw.seek(header.offset)
        while True:
            chunk = raw.read(chunk_bytes)
            if not chunk:
                break

            chunk = carry + chunk
            if len(chunk) % 2:
                carry = chunk[-1:]
                chunk = chunk[:-1]
            else:
                carry = b""
            if not chunk:
                continue

            words = np.frombuffer(chunk, dtype="<u2")
            stats.words += int(words.size)
            stats.type_counts += np.bincount((words >> 12).astype(np.int64), minlength=16)

            for word_value in words:
                word = int(word_value)
                event_type = word >> 12
                payload = word & 0x0FFF

                if event_type == 0x2:
                    note_events()
                elif event_type == 0x4:
                    note_events(int((word & 0x0FFF).bit_count()))
                elif event_type == 0x5:
                    note_events(int((word & 0x00FF).bit_count()))
                elif event_type == 0x6:
                    current_ts = (time_high << 12) | payload
                elif event_type == 0x8:
                    high_12 = payload
                    if (
                        last_time_high_12 is not None
                        and high_12 < last_time_high_12
                        and last_time_high_12 - high_12 > 2048
                    ):
                        stats.time_high_wraps += 1
                    last_time_high_12 = high_12
                    time_high = (stats.time_high_wraps << 12) | high_12

    if carry:
        stats.ignored_trailing_bytes += len(carry)
    return stats


def iter_evt3_events(
    header: RawHeader,
    *,
    start_us: int,
    duration_us: int | None,
    max_events: int | None,
    chunk_bytes: int,
    handler: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None],
) -> DecodeStats:
    stats = DecodeStats()
    current_y = 0
    current_x = 0
    current_p = 0
    time_high = 0
    last_time_high_12 = None
    current_ts = 0
    first_ts = None
    end_us = None if duration_us is None else start_us + duration_us
    carry = b""

    xs: list[int] = []
    ys: list[int] = []
    ps: list[int] = []
    ts_values: list[int] = []
    rel_values: list[int] = []
    batch_size = 100_000

    def flush() -> None:
        if not xs:
            return
        xs_arr = np.asarray(xs, dtype=np.int32)
        ys_arr = np.asarray(ys, dtype=np.int32)
        ps_arr = np.asarray(ps, dtype=np.uint8)
        ts_arr = np.asarray(ts_values, dtype=np.int64)
        rel_arr = np.asarray(rel_values, dtype=np.int64)
        stats.update_events(xs_arr, ys_arr, ps_arr, ts_arr, rel_arr, header.width, header.height)
        handler(xs_arr, ys_arr, ps_arr, ts_arr, rel_arr)
        xs.clear()
        ys.clear()
        ps.clear()
        ts_values.clear()
        rel_values.clear()

    def emit(x: int, y: int, p: int, t: int) -> None:
        nonlocal first_ts

        stats.decoded_events += 1
        stats.first_ts_us = t if stats.first_ts_us is None else stats.first_ts_us
        stats.last_ts_us = t

        if first_ts is None:
            first_ts = t

        rel = t - first_ts
        if rel < start_us:
            return
        if end_us is not None and rel >= end_us:
            stats.stop_reason = "duration"
            raise StopDecoding
        if max_events is not None and stats.kept_events + len(xs) >= max_events:
            stats.stop_reason = "max-events"
            raise StopDecoding

        xs.append(x)
        ys.append(y)
        ps.append(p)
        ts_values.append(t)
        rel_values.append(rel - start_us)
        if len(xs) >= batch_size:
            flush()

    try:
        with header.path.open("rb") as raw:
            raw.seek(header.offset)
            while True:
                chunk = raw.read(chunk_bytes)
                if not chunk:
                    break

                chunk = carry + chunk
                if len(chunk) % 2:
                    carry = chunk[-1:]
                    chunk = chunk[:-1]
                else:
                    carry = b""
                if not chunk:
                    continue

                words = np.frombuffer(chunk, dtype="<u2")
                stats.words += int(words.size)
                stats.type_counts += np.bincount((words >> 12).astype(np.int64), minlength=16)

                for word_value in words:
                    word = int(word_value)
                    event_type = word >> 12
                    payload = word & 0x0FFF

                    if event_type == 0x0:
                        current_y = word & 0x07FF
                    elif event_type == 0x2:
                        emit(word & 0x07FF, current_y, (word >> 11) & 0x1, current_ts)
                    elif event_type == 0x3:
                        current_x = word & 0x07FF
                        current_p = (word >> 11) & 0x1
                    elif event_type == 0x4:
                        valid = payload
                        for bit in range(12):
                            if valid & (1 << bit):
                                emit(current_x + bit, current_y, current_p, current_ts)
                        current_x += 12
                    elif event_type == 0x5:
                        valid = word & 0x00FF
                        for bit in range(8):
                            if valid & (1 << bit):
                                emit(current_x + bit, current_y, current_p, current_ts)
                        current_x += 8
                    elif event_type == 0x6:
                        current_ts = (time_high << 12) | payload
                    elif event_type == 0x8:
                        high_12 = payload
                        if (
                            last_time_high_12 is not None
                            and high_12 < last_time_high_12
                            and last_time_high_12 - high_12 > 2048
                        ):
                            stats.time_high_wraps += 1
                        last_time_high_12 = high_12
                        time_high = (stats.time_high_wraps << 12) | high_12
    except StopDecoding:
        pass

    if carry:
        stats.ignored_trailing_bytes += len(carry)
    flush()
    return stats


def save_event_image(pos: np.ndarray, neg: np.ndarray, path: Path) -> None:
    image = np.zeros((pos.shape[0], pos.shape[1], 3), dtype=np.uint8)
    nonzero_counts = np.concatenate((pos[pos > 0], neg[neg > 0]))
    if nonzero_counts.size == 0:
        cv2.imwrite(str(path), image)
        return

    scale = max(float(np.percentile(nonzero_counts, 99.5)), 1.0)
    pos_norm = np.clip(np.log1p(pos.astype(np.float32)) / np.log1p(scale), 0.0, 1.0)
    neg_norm = np.clip(np.log1p(neg.astype(np.float32)) / np.log1p(scale), 0.0, 1.0)
    activity = np.maximum(pos_norm, neg_norm)

    # OpenCV writes BGR: OFF events are blue/cyan, ON events are red/yellow.
    image[..., 0] = np.clip(255 * neg_norm, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip(110 * activity, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip(255 * pos_norm, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def print_header(header: RawHeader) -> None:
    print(f"File: {header.path}")
    print(f"Size: {header.path.stat().st_size:,} bytes")
    print(f"Header bytes: {header.offset:,}")
    print(f"Format: {header.format_name}")
    print(f"Geometry: {header.width}x{header.height}")
    for key in ("camera_integrator_name", "date", "generation", "sensor_generation", "sensor_name", "serial_number"):
        if key in header.metadata:
            print(f"{key}: {header.metadata[key]}")


def print_stats(stats: DecodeStats, bin_us: int) -> None:
    print("\nDecoded events")
    print(f"16-bit words read: {stats.words:,}")
    print(f"Stop reason: {stats.stop_reason}")
    if stats.ignored_trailing_bytes:
        print(f"Ignored trailing byte(s): {stats.ignored_trailing_bytes}")
    print(f"Decoded CD events before stop: {stats.decoded_events:,}")
    print(f"Events kept for outputs: {stats.kept_events:,}")

    if stats.first_ts_us is not None and stats.last_ts_us is not None:
        duration_s = max((stats.last_ts_us - stats.first_ts_us) / 1_000_000.0, 0.0)
        print(f"Raw timestamp range read: {stats.first_ts_us:,} to {stats.last_ts_us:,} us ({duration_s:.3f} s)")
    if stats.kept_first_rel_us is not None and stats.kept_last_rel_us is not None:
        kept_s = max((stats.kept_last_rel_us - stats.kept_first_rel_us) / 1_000_000.0, bin_us / 1_000_000.0)
        print(f"Converted span: {stats.kept_first_rel_us:,} to {stats.kept_last_rel_us:,} us")
        print(f"Average converted event rate: {stats.kept_events / kept_s:,.0f} events/s")

    print(f"Polarity: ON={stats.on_events:,}, OFF={stats.off_events:,}")
    if stats.min_x is not None:
        print(f"Observed x range: {stats.min_x}..{stats.max_x}")
        print(f"Observed y range: {stats.min_y}..{stats.max_y}")
    if stats.out_of_bounds_events:
        print(f"Out-of-bounds events: {stats.out_of_bounds_events:,}")
    if stats.time_high_wraps:
        print(f"Timestamp high wraps detected: {stats.time_high_wraps}")

    print("\nEVT3 word types read")
    for event_type, count in enumerate(stats.type_counts):
        if count:
            name = EVT_TYPES.get(event_type, "RESERVED")
            print(f"  0x{event_type:X} {name}: {int(count):,}")


def resolve_input_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)

    raw_files = sorted(DEFAULT_DATASET_DIR.glob("*.raw"))
    if not raw_files:
        raise FileNotFoundError(f"No .raw files found in {DEFAULT_DATASET_DIR}")

    print("No file provided. Found recordings:")
    for idx, path in enumerate(raw_files):
        print(f"  [{idx}] {path} ({path.stat().st_size:,} bytes)")
    print(f"Using first recording by default: {raw_files[0]}\n")
    return raw_files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Prophesee EVT3 .raw event-camera recordings into event-frame PNGs."
    )
    parser.add_argument("raw_file", nargs="?", help="Path to a .raw file. Defaults to first file in Datasets/Recordings_from_matrice.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Parent directory for the converted frame folder.")
    parser.add_argument("--start-ms", type=float, default=0.0, help="Conversion start time relative to first decoded event.")
    parser.add_argument("--duration-ms", type=float, default=None, help="Optional conversion duration. Omit to convert all remaining events.")
    parser.add_argument("--frame-ms", type=float, default=DEFAULT_FRAME_MS, help="Duration accumulated into each output frame.")
    parser.add_argument("--scan-all", action="store_true", help="Compatibility flag; full-file conversion is now the default when --duration-ms is omitted.")
    parser.add_argument("--chunk-mb", type=int, default=16, help="Streaming read chunk size in MiB.")
    parser.add_argument("--info-only", action="store_true", help="Only print header information; do not decode events.")
    return parser


def make_unique_output_dir(parent_dir: Path, stem: str) -> Path:
    base_dir = parent_dir / f"{stem}_event_frames"
    if not base_dir.exists():
        return base_dir

    for idx in range(1, 1000):
        candidate = parent_dir / f"{stem}_event_frames_{idx:03d}"
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Could not find an unused output folder under {parent_dir}")


def main() -> int:
    args = build_parser().parse_args()
    raw_path = resolve_input_path("/home/daniel/Optical_Flow/Datasets/Recordings_from_matrice/20251009_224707.raw")
    header = parse_header(raw_path)
    print_header(header)

    if "EVT3" not in header.format_name:
        print("\nThis script currently decodes EVT3 only. The header format is not EVT3.")
        return 2
    if args.info_only:
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_us = int(args.start_ms * 1000.0)
    frame_us = max(int(args.frame_ms * 1000.0), 1)
    chunk_bytes = max(args.chunk_mb, 1) * 1024 * 1024

    if args.duration_ms is None:
        print("\nScanning timestamps to compute total frame count...")
        bounds = scan_evt3_time_bounds(header, chunk_bytes=chunk_bytes)
        if bounds.first_ts_us is None or bounds.last_ts_us is None:
            print("No contrast-detection events were found in this recording.")
            return 1
        recording_duration_us = bounds.last_ts_us - bounds.first_ts_us + 1
        if start_us >= recording_duration_us:
            print(f"--start-ms is beyond the recording duration ({recording_duration_us / 1000.0:.3f} ms).")
            return 1
        duration_us = recording_duration_us - start_us
    else:
        duration_us = max(int(args.duration_ms * 1000.0), 1)

    total_frames = max(int(math.ceil(duration_us / frame_us)), 1)
    frames_dir = make_unique_output_dir(output_dir, raw_path.stem)
    print(f"\nFrame duration: {frame_us / 1000.0:.3f} ms")
    print(f"Total frames: {total_frames:,}")
    print(f"Writing frames to: {frames_dir}")
    sys.stdout.flush()

    writer = EventFrameWriter(
        width=header.width,
        height=header.height,
        frame_us=frame_us,
        output_dir=frames_dir,
        total_frames=total_frames,
        progress=ProgressBar("Converting", total_frames),
    )

    def handle_batch(
        xs: np.ndarray,
        ys: np.ndarray,
        ps: np.ndarray,
        _ts: np.ndarray,
        rel_ts: np.ndarray,
    ) -> None:
        writer.add(xs, ys, ps, rel_ts)

    stats = iter_evt3_events(
        header,
        start_us=start_us,
        duration_us=duration_us,
        max_events=None,
        chunk_bytes=chunk_bytes,
        handler=handle_batch,
    )
    writer.finalize()
    print_stats(stats, frame_us)

    print("\nOutputs")
    print(f"Event frames folder: {frames_dir}")
    print(f"Frames written: {writer.current_frame:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
