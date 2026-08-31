#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
RUNS_ROOT=${RUNS_ROOT:-$SWEEP_ROOT/training_runs}
LOG_DIR=${LOG_DIR:-$RUNS_ROOT/dropout_search_logs}
BASELINE_RUN=${BASELINE_RUN:-12}

if (($# > 0)); then
  VALUES=("$@")
else
  VALUES=(0.1 0.2 0.4)
fi

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/dropout_search.log"
}

summarize() {
  python3 - "$RUNS_ROOT" "$BASELINE_RUN" <<'PY' || true
import json
from pathlib import Path
import sys

runs_root = Path(sys.argv[1])
baseline_run = sys.argv[2]
rows = []
for run_dir in sorted(runs_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
    if not run_dir.name.isdigit():
        continue
    if run_dir.name != baseline_run:
        log_path = run_dir / "logs" / "dropout_search_wrapper.log"
        if not log_path.exists():
            continue
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    if data.get("split_strategy") != "temporal_window_fraction":
        continue
    if data.get("val_fraction") != 0.2:
        continue
    if data.get("l2_lambda") != 1e-5:
        continue
    rows.append(
        {
            "run": run_dir.name,
            "dropout": data.get("dropout", 0.0),
            "l2_lambda": data.get("l2_lambda"),
            "best_epoch": data.get("best_epoch"),
            "val_rmse": data.get("final_val", {}).get("rmse"),
            "val_mae": data.get("final_val", {}).get("mae"),
            "train_rmse": data.get("final_train", {}).get("rmse"),
            "train_mae": data.get("final_train", {}).get("mae"),
        }
    )

out = runs_root / "temporal_80_20_dropout_search.md"
lines = [
    "# Temporal 80/20 Dropout Search",
    "",
    "| Run | Dropout | L2 lambda | Best epoch | Val RMSE | Val MAE | Train RMSE | Train MAE |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['run']} | {row['dropout']} | {row['l2_lambda']} | {row['best_epoch']} | "
        f"{row['val_rmse']:.6f} | {row['val_mae']:.6f} | "
        f"{row['train_rmse']:.6f} | {row['train_mae']:.6f} |"
    )
out.write_text("\n".join(lines) + "\n")
print(out)
PY
}

main() {
  cd "$ROOT"
  log "temporal 80/20 dropout search started"
  log "baseline run: $BASELINE_RUN"
  log "values: ${VALUES[*]}"
  summarize | tee -a "$LOG_DIR/dropout_search.log"

  local value
  for value in "${VALUES[@]}"; do
    local run_dir
    run_dir=$(ROOT="$ROOT" RUNS_ROOT="$RUNS_ROOT" bash "$ROOT/run_scripts/next_evimo_training_run_dir.sh")
    mkdir -p "$run_dir/logs"
    log "starting dropout=$value in $run_dir"
    RUN_DIR="$run_dir" L2_LAMBDA=1e-5 WEIGHT_DECAY=0 DROPOUT="$value" bash "$ROOT/run_scripts/train_evimo_temporal_80_20_node2.sh" \
      2>&1 | tee "$run_dir/logs/dropout_search_wrapper.log"
    log "finished dropout=$value in $run_dir"
    summarize | tee -a "$LOG_DIR/dropout_search.log"
  done

  log "temporal 80/20 dropout search done"
}

main "$@"
