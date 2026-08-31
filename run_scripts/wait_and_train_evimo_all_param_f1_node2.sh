#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
PYTHON=${PYTHON:-$ROOT/E-MoFlow/.venv/bin/python}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
COMBINED_ROOT=${COMBINED_ROOT:-$SWEEP_ROOT/all_train_prefixed}
OUTPUT_DIR=${OUTPUT_DIR:-$(ROOT="$ROOT" RUNS_ROOT="$SWEEP_ROOT/training_runs" bash "$ROOT/run_scripts/next_evimo_training_run_dir.sh")}
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/logs}
POLL_SECONDS=${POLL_SECONDS:-300}

BOX_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05 seq_06 seq_07 seq_08 seq_09 seq_10 seq_11)
FLOOR_SEQS=(seq_00 seq_01 seq_02)
WALL_SEQS=(seq_00 seq_01 seq_02)
VAL_SEQUENCES=(box_seq_11 floor_seq_02 wall_seq_02)

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/scheduler.log"
}

sequence_ready() {
  local group=$1
  local seq=$2
  local dir="$SWEEP_ROOT/$group/$seq"
  test -f "$dir/grid_metrics.csv" &&
    test -f "$dir/f1_matrix.md" &&
    test -f "$dir/metadata.json" &&
    test -d "$dir/window_inputs"
}

wait_for_sequences() {
  while true; do
    local missing=()
    local seq
    for seq in "${BOX_SEQS[@]}"; do
      sequence_ready box "$seq" || missing+=("box/$seq")
    done
    for seq in "${FLOOR_SEQS[@]}"; do
      sequence_ready floor "$seq" || missing+=("floor/$seq")
    done
    for seq in "${WALL_SEQS[@]}"; do
      sequence_ready wall "$seq" || missing+=("wall/$seq")
    done

    if ((${#missing[@]} == 0)); then
      log "all sweep inputs are ready"
      return
    fi

    log "waiting for ${#missing[@]} sequence(s): ${missing[*]}"
    sleep "$POLL_SECONDS"
  done
}

link_sequences() {
  mkdir -p "$COMBINED_ROOT"
  local seq
  for seq in "${BOX_SEQS[@]}"; do
    ln -sfn "$SWEEP_ROOT/box/$seq" "$COMBINED_ROOT/box_$seq"
  done
  for seq in "${FLOOR_SEQS[@]}"; do
    ln -sfn "$SWEEP_ROOT/floor/$seq" "$COMBINED_ROOT/floor_$seq"
  done
  for seq in "${WALL_SEQS[@]}"; do
    ln -sfn "$SWEEP_ROOT/wall/$seq" "$COMBINED_ROOT/wall_$seq"
  done
}

csv_join() {
  local IFS=,
  printf '%s' "$*"
}

make_plots() {
  "$PYTHON" - "$OUTPUT_DIR" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
plots = out / "plots"
plots.mkdir(exist_ok=True)

with (out / "history.csv").open(newline="", encoding="utf-8") as handle:
    history = list(csv.DictReader(handle))

epochs = [int(row["epoch"]) for row in history]
train = [float(row["train_rmse"]) for row in history]
val = [float(row["val_rmse"]) for row in history]
best_idx = min(range(len(val)), key=val.__getitem__)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs, train, marker="o", linewidth=1.5, label="train")
ax.plot(epochs, val, marker="o", linewidth=1.5, label="validation")
ax.scatter([epochs[best_idx]], [val[best_idx]], color="black", zorder=3, label=f"best val epoch {epochs[best_idx]}")
ax.set_xlabel("epoch")
ax.set_ylabel("RMSE")
ax.set_title("F1 predictor RMSE over epochs")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(plots / "rmse_over_epochs.png", dpi=180)
plt.close(fig)

by_seq = defaultdict(lambda: [0.0, 0])
with (out / "predictions.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        err = float(row["pred_f1"]) - float(row["f1"])
        bucket = by_seq[(row["split"], row["sequence"])]
        bucket[0] += err * err
        bucket[1] += 1

items = sorted((split, seq, math.sqrt(total / count)) for (split, seq), (total, count) in by_seq.items() if count)
labels = [f"{split}:{seq}" for split, seq, _ in items]
values = [rmse for _, _, rmse in items]
colors = ["#4c78a8" if split == "train" else "#f58518" for split, _, _ in items]

fig_w = max(9, 0.45 * len(labels))
fig, ax = plt.subplots(figsize=(fig_w, 4.8))
ax.bar(labels, values, color=colors)
ax.set_ylabel("RMSE")
ax.set_title("Checkpoint RMSE by sequence")
ax.tick_params(axis="x", rotation=65)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(plots / "rmse_by_sequence.png", dpi=180)
plt.close(fig)

(plots / "README.md").write_text(
    "# RMSE Plots\n\n"
    "- `rmse_over_epochs.png`: train and validation RMSE across 100 epochs.\n"
    "- `rmse_by_sequence.png`: checkpoint RMSE grouped by split and sequence.\n"
    f"\nBest validation epoch by RMSE: {epochs[best_idx]} ({val[best_idx]:.4f}).\n",
    encoding="utf-8",
)
PY
}

main() {
  cd "$ROOT"
  log "node2 post-sweep training watcher started"
  log "output: $OUTPUT_DIR"
  if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    log "summary already exists; not starting another training run"
    exit 0
  fi

  wait_for_sequences
  link_sequences

  local all_sequences val_sequences
  all_sequences=$(csv_join \
    box_seq_00 box_seq_01 box_seq_02 box_seq_03 box_seq_04 box_seq_05 box_seq_06 box_seq_07 box_seq_08 box_seq_09 box_seq_10 box_seq_11 \
    floor_seq_00 floor_seq_01 floor_seq_02 \
    wall_seq_00 wall_seq_01 wall_seq_02)
  val_sequences=$(csv_join "${VAL_SEQUENCES[@]}")

  log "starting 100-epoch training on node2"
  "$PYTHON" EvMotionSeg/tools/train_evimo_multi_sequence_param_f1_predictor.py \
    --sweep-root "$COMBINED_ROOT" \
    --sequences "$all_sequences" \
    --val-sequences "$val_sequences" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 0 \
    --device auto \
    2>&1 | tee "$LOG_DIR/train.log"

  log "training finished; generating RMSE plots"
  make_plots 2>&1 | tee "$LOG_DIR/plots.log"
  log "done"
}

main "$@"
