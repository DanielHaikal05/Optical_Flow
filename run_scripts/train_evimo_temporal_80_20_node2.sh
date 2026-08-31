#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
PYTHON=${PYTHON:-$ROOT/E-MoFlow/.venv/bin/python}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
RUNS_ROOT=${RUNS_ROOT:-$SWEEP_ROOT/training_runs}
COMBINED_ROOT=${COMBINED_ROOT:-$SWEEP_ROOT/all_train_prefixed}
L2_LAMBDA=${L2_LAMBDA:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
DROPOUT=${DROPOUT:-0}
ENCODER_DEPTH=${ENCODER_DEPTH:-2}
ENCODER_VARIANT=${ENCODER_VARIANT:-standard}
MLP_HIDDEN_DIMS=${MLP_HIDDEN_DIMS:-32,16}
PCA_COMPONENTS=${PCA_COMPONENTS:-0}
EPOCHS=${EPOCHS:-100}
LR_SCHEDULER=${LR_SCHEDULER:-none}
LR_SCHEDULER_PATIENCE=${LR_SCHEDULER_PATIENCE:-5}
LR_SCHEDULER_FACTOR=${LR_SCHEDULER_FACTOR:-0.5}
MIN_LR=${MIN_LR:-0}
INCLUDE_TABLETOP=${INCLUDE_TABLETOP:-0}
LOSS_TYPE=${LOSS_TYPE:-mse}
TARGET_MODE=${TARGET_MODE:-raw}
HUBER_DELTA=${HUBER_DELTA:-0.1}
WEIGHTED_TEMPERATURE=${WEIGHTED_TEMPERATURE:-0.1}
WEIGHTED_ALPHA=${WEIGHTED_ALPHA:-0.5}
RANKING_TEMPERATURE=${RANKING_TEMPERATURE:-0.1}
RANKING_LAMBDA=${RANKING_LAMBDA:-0.1}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}

BOX_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05 seq_06 seq_07 seq_08 seq_09 seq_10 seq_11)
FLOOR_SEQS=(seq_00 seq_01 seq_02)
WALL_SEQS=(seq_00 seq_01 seq_02)
TABLE_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05)
TABLETOP_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05)

RUN_DIR=${RUN_DIR:-$(ROOT="$ROOT" RUNS_ROOT="$RUNS_ROOT" bash "$ROOT/run_scripts/next_evimo_training_run_dir.sh")}
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/temporal_80_20_pipeline.log"
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

check_ready() {
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
  for seq in "${TABLE_SEQS[@]}"; do
    sequence_ready table "$seq" || missing+=("table/$seq")
  done
  if [ "$INCLUDE_TABLETOP" = "1" ]; then
    for seq in "${TABLETOP_SEQS[@]}"; do
      sequence_ready tabletop "$seq" || missing+=("tabletop/$seq")
    done
  fi

  if ((${#missing[@]} > 0)); then
    printf 'missing required sequence output(s): %s\n' "${missing[*]}" >&2
    return 1
  fi
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
  for seq in "${TABLE_SEQS[@]}"; do
    ln -sfn "$SWEEP_ROOT/table/$seq" "$COMBINED_ROOT/table_$seq"
  done
  if [ "$INCLUDE_TABLETOP" = "1" ]; then
    for seq in "${TABLETOP_SEQS[@]}"; do
      ln -sfn "$SWEEP_ROOT/tabletop/$seq" "$COMBINED_ROOT/tabletop_$seq"
    done
  fi
}

csv_join() {
  local IFS=,
  printf '%s' "$*"
}

make_plots() {
  "$PYTHON" - "$RUN_DIR" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
run_name = out.name
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
ax.set_title(f"Run {run_name} RMSE over epochs")
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
fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(labels)), 4.8))
ax.bar(labels, values, color=colors)
ax.set_ylabel("RMSE")
ax.set_title(f"Run {run_name} checkpoint RMSE by sequence")
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
  log "temporal 80/20 training pipeline started"
  log "run dir: $RUN_DIR"
  printf '%s\n' "$RUN_DIR" > "$SWEEP_ROOT/latest_training_dir.txt"

  check_ready
  link_sequences

  local all_sequences
  local sequence_names=(
    box_seq_00 box_seq_01 box_seq_02 box_seq_03 box_seq_04 box_seq_05 box_seq_06 box_seq_07 box_seq_08 box_seq_09 box_seq_10 box_seq_11
    floor_seq_00 floor_seq_01 floor_seq_02
    wall_seq_00 wall_seq_01 wall_seq_02
    table_seq_00 table_seq_01 table_seq_02 table_seq_03 table_seq_04 table_seq_05
  )
  if [ "$INCLUDE_TABLETOP" = "1" ]; then
    sequence_names+=(tabletop_seq_00 tabletop_seq_01 tabletop_seq_02 tabletop_seq_03 tabletop_seq_04 tabletop_seq_05)
  fi
  all_sequences=$(csv_join "${sequence_names[@]}")

  log "starting temporal 80/20 run: $EPOCHS epochs, sequences=${#sequence_names[@]}, encoder_depth=$ENCODER_DEPTH, encoder_variant=$ENCODER_VARIANT, mlp_hidden_dims=$MLP_HIDDEN_DIMS, pca_components=$PCA_COMPONENTS, l2_lambda=$L2_LAMBDA, weight_decay=$WEIGHT_DECAY, dropout=$DROPOUT, loss_type=$LOSS_TYPE, target_mode=$TARGET_MODE, lr_scheduler=$LR_SCHEDULER, lr_scheduler_patience=$LR_SCHEDULER_PATIENCE, lr_scheduler_factor=$LR_SCHEDULER_FACTOR, early_stopping_patience=$EARLY_STOPPING_PATIENCE"
  "$PYTHON" EvMotionSeg/tools/train_evimo_multi_sequence_param_f1_predictor.py \
    --sweep-root "$COMBINED_ROOT" \
    --sequences "$all_sequences" \
    --split-strategy temporal_window_fraction \
    --val-fraction 0.2 \
    --shuffle-split-indices \
    --output-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --batch-size 128 \
    --num-workers 0 \
    --device auto \
    --encoder-depth "$ENCODER_DEPTH" \
    --encoder-variant "$ENCODER_VARIANT" \
    --mlp-hidden-dims "$MLP_HIDDEN_DIMS" \
    --pca-components "$PCA_COMPONENTS" \
    --dropout "$DROPOUT" \
    --weight-decay "$WEIGHT_DECAY" \
    --l2-lambda "$L2_LAMBDA" \
    --lr-scheduler "$LR_SCHEDULER" \
    --lr-scheduler-patience "$LR_SCHEDULER_PATIENCE" \
    --lr-scheduler-factor "$LR_SCHEDULER_FACTOR" \
    --min-lr "$MIN_LR" \
    --loss-type "$LOSS_TYPE" \
    --target-mode "$TARGET_MODE" \
    --huber-delta "$HUBER_DELTA" \
    --weighted-temperature "$WEIGHTED_TEMPERATURE" \
    --weighted-alpha "$WEIGHTED_ALPHA" \
    --ranking-temperature "$RANKING_TEMPERATURE" \
    --ranking-lambda "$RANKING_LAMBDA" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    2>&1 | tee "$LOG_DIR/train.log"

  log "training finished; generating RMSE plots"
  make_plots 2>&1 | tee "$LOG_DIR/plots.log"

  log "generating run GT/training document"
  "$PYTHON" EvMotionSeg/tools/make_evimo_train_gt_document.py \
    --sweep-root "$SWEEP_ROOT" \
    --training-output "$RUN_DIR" \
    --output "$RUN_DIR/gt_document.md" \
    2>&1 | tee "$LOG_DIR/gt_document.log"
  log "done"
}

main "$@"
