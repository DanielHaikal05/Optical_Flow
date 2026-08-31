#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
PYTHON=${PYTHON:-$ROOT/E-MoFlow/.venv/bin/python}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
EVIMO2_ROOT=${EVIMO2_ROOT:-$ROOT/Datasets/EVIMO2_left/sanity/sliding}
EVIMO2_OUT_ROOT=${EVIMO2_OUT_ROOT:-$SWEEP_ROOT/evimo2_left}
RUNS_ROOT=${RUNS_ROOT:-$SWEEP_ROOT/training_runs}
COMBINED_ROOT=${COMBINED_ROOT:-$SWEEP_ROOT/all_train_evimo_evimo2_prefixed}
BUILD_DIR=${EVMOTIONSEG_BUILD_DIR:-/tmp/evimo2_param_sweep_build}
SAMPLE_COUNT=${EVIMO2_SWEEP_SAMPLE_COUNT:-30}
SAMPLE_SEED=${EVIMO2_SWEEP_SAMPLE_SEED:-20260818}
RANDOM_SEED=${EVMOTIONSEG_RANDOM_SEED:-20260818}
MAX_JOBS=${EVIMO2_SWEEP_MAX_JOBS:-1}
SEQUENCES=${EVIMO2_SWEEP_SEQUENCES:-sliding_00_000000 sliding_01_000000}

BOX_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05 seq_06 seq_07 seq_08 seq_09 seq_10 seq_11)
FLOOR_SEQS=(seq_00 seq_01 seq_02)
WALL_SEQS=(seq_00 seq_01 seq_02)
TABLE_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05)
TABLETOP_SEQS=(seq_00 seq_01 seq_02 seq_03 seq_04 seq_05)

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

sequence_ready() {
  local dir=$1
  test -f "$dir/grid_metrics.csv" &&
    test -f "$dir/f1_matrix.md" &&
    test -f "$dir/metadata.json" &&
    test -d "$dir/window_inputs"
}

link_one() {
  local source=$1
  local dest=$2
  ln -sfn "$source" "$dest"
}

check_evimo_ready() {
  local missing=()
  local seq
  for seq in "${BOX_SEQS[@]}"; do sequence_ready "$SWEEP_ROOT/box/$seq" || missing+=("box/$seq"); done
  for seq in "${FLOOR_SEQS[@]}"; do sequence_ready "$SWEEP_ROOT/floor/$seq" || missing+=("floor/$seq"); done
  for seq in "${WALL_SEQS[@]}"; do sequence_ready "$SWEEP_ROOT/wall/$seq" || missing+=("wall/$seq"); done
  for seq in "${TABLE_SEQS[@]}"; do sequence_ready "$SWEEP_ROOT/table/$seq" || missing+=("table/$seq"); done
  for seq in "${TABLETOP_SEQS[@]}"; do sequence_ready "$SWEEP_ROOT/tabletop/$seq" || missing+=("tabletop/$seq"); done
  if ((${#missing[@]} > 0)); then
    printf 'missing required EVIMO output(s): %s\n' "${missing[*]}" >&2
    return 1
  fi
}

run_evimo2_gt() {
  mkdir -p "$EVIMO2_OUT_ROOT/logs"
  "$ROOT/EvMotionSeg/tools/build_standalone_portable.sh" "$BUILD_DIR"
  read -r -a seqs <<< "$SEQUENCES"
  printf '%s\n' "${seqs[@]}" > "$EVIMO2_OUT_ROOT/sequences.txt"
  local active=0
  local failures=0
  local seq
  for seq in "${seqs[@]}"; do
    local out_dir="$EVIMO2_OUT_ROOT/$seq"
    mkdir -p "$out_dir/logs"
    if sequence_ready "$out_dir"; then
      log "EVIMO2 GT already ready: $seq"
      continue
    fi
    (
      set -euo pipefail
      log "Starting EVIMO2 GT $seq"
      "$PYTHON" "$ROOT/EvMotionSeg/tools/sweep_evimo2_scene_terms.py" \
        --sequence-dir "$EVIMO2_ROOT/$seq" \
        --output-dir "$out_dir" \
        --binary "$BUILD_DIR/motion_segmentation_standalone" \
        --sample-count "$SAMPLE_COUNT" \
        --sample-seed "$SAMPLE_SEED" \
        --random-seed "$RANDOM_SEED" \
        --training-set EVIMO \
        --ensemble 3 \
        --no-auto-scale-time \
        --cleanup-run-dirs \
        --resume
      log "Finished EVIMO2 GT $seq"
    ) > "$out_dir/logs/driver.log" 2>&1 &
    echo "$!" > "$out_dir/job.pid"
    active=$((active + 1))
    if (( active >= MAX_JOBS )); then
      if ! wait -n; then failures=$((failures + 1)); fi
      active=$((active - 1))
    fi
  done
  while (( active > 0 )); do
    if ! wait -n; then failures=$((failures + 1)); fi
    active=$((active - 1))
  done
  log "EVIMO2 GT finished with $failures failed sequence job(s)." | tee "$EVIMO2_OUT_ROOT/logs/batch.done"
  return "$failures"
}

build_combined_root() {
  check_evimo_ready
  mkdir -p "$COMBINED_ROOT"
  local seq
  local sequence_names=()
  for seq in "${BOX_SEQS[@]}"; do link_one "$SWEEP_ROOT/box/$seq" "$COMBINED_ROOT/box_$seq"; sequence_names+=("box_$seq"); done
  for seq in "${FLOOR_SEQS[@]}"; do link_one "$SWEEP_ROOT/floor/$seq" "$COMBINED_ROOT/floor_$seq"; sequence_names+=("floor_$seq"); done
  for seq in "${WALL_SEQS[@]}"; do link_one "$SWEEP_ROOT/wall/$seq" "$COMBINED_ROOT/wall_$seq"; sequence_names+=("wall_$seq"); done
  for seq in "${TABLE_SEQS[@]}"; do link_one "$SWEEP_ROOT/table/$seq" "$COMBINED_ROOT/table_$seq"; sequence_names+=("table_$seq"); done
  for seq in "${TABLETOP_SEQS[@]}"; do link_one "$SWEEP_ROOT/tabletop/$seq" "$COMBINED_ROOT/tabletop_$seq"; sequence_names+=("tabletop_$seq"); done
  read -r -a evimo2_seqs <<< "$SEQUENCES"
  for seq in "${evimo2_seqs[@]}"; do
    sequence_ready "$EVIMO2_OUT_ROOT/$seq" || {
      printf 'missing EVIMO2 output: %s\n' "$EVIMO2_OUT_ROOT/$seq" >&2
      return 1
    }
    link_one "$EVIMO2_OUT_ROOT/$seq" "$COMBINED_ROOT/evimo2_$seq"
    sequence_names+=("evimo2_$seq")
  done
  local IFS=,
  printf '%s' "${sequence_names[*]}" > "$EVIMO2_OUT_ROOT/combined_sequences.csv"
}

train_config() {
  local run=$1
  local variant=$2
  local pca_components=$3
  local run_dir="$RUNS_ROOT/$run"
  local sequence_csv
  sequence_csv=$(cat "$EVIMO2_OUT_ROOT/combined_sequences.csv")
  mkdir -p "$run_dir/logs"
  log "Starting training run $run variant=$variant pca_components=$pca_components"
  "$PYTHON" "$ROOT/EvMotionSeg/tools/train_evimo_multi_sequence_param_f1_predictor.py" \
    --sweep-root "$COMBINED_ROOT" \
    --sequences "$sequence_csv" \
    --split-strategy temporal_window_fraction \
    --val-fraction 0.2 \
    --shuffle-split-indices \
    --output-dir "$run_dir" \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 0 \
    --device auto \
    --encoder-depth 2 \
    --encoder-variant "$variant" \
    --mlp-hidden-dims 32,16 \
    --pca-components "$pca_components" \
    --dropout 0.1 \
    --weight-decay 0 \
    --l2-lambda 1e-4 \
    --lr-scheduler none \
    --lr-scheduler-patience 5 \
    --lr-scheduler-factor 0.5 \
    --min-lr 0 \
    --loss-type mse \
    --target-mode raw \
    --early-stopping-patience 0 \
    2>&1 | tee "$run_dir/logs/train.log"
  log "Finished training run $run"
}

make_plots() {
  local run=$1
  local run_dir="$RUNS_ROOT/$run"
"$PYTHON" - "$run_dir" <<'PY'
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
history = list(csv.DictReader((out / "history.csv").open(newline="", encoding="utf-8")))
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
ax.set_title(f"Run {out.name} RMSE over epochs")
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
ax.set_title(f"Run {out.name} checkpoint RMSE by sequence")
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
  mkdir -p "$EVIMO2_OUT_ROOT/logs" "$RUNS_ROOT"
  log "EVIMO2 GT + config 28/42 training pipeline started"
  run_evimo2_gt
  build_combined_root
  train_config 64 grid49_pool3x4 0
  make_plots 64 2>&1 | tee "$RUNS_ROOT/64/logs/plots.log"
  train_config 65 pca49_pool2x3 4
  make_plots 65 2>&1 | tee "$RUNS_ROOT/65/logs/plots.log"
  log "EVIMO2 GT + config 28/42 training pipeline finished"
}

main "$@"
