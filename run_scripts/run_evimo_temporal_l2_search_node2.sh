#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
RUNS_ROOT=${RUNS_ROOT:-$SWEEP_ROOT/training_runs}
LOG_DIR=${LOG_DIR:-$RUNS_ROOT/l2_search_logs}
if (($# > 0)); then
  VALUES=("$@")
else
  VALUES=(0 1e-5)
fi

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/l2_search.log"
}

summarize() {
  python3 - "$RUNS_ROOT" <<'PY' || true
import json
from pathlib import Path
import sys

runs_root = Path(sys.argv[1])
rows = []
for run_dir in sorted(runs_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
    if not run_dir.name.isdigit():
        continue
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    if data.get("split_strategy") != "temporal_window_fraction":
        continue
    if data.get("val_fraction") != 0.2:
        continue
    rows.append(
        {
            "run": run_dir.name,
            "l2_lambda": data.get("l2_lambda"),
            "best_epoch": data.get("best_epoch"),
            "val_rmse": data.get("final_val", {}).get("rmse"),
            "val_mae": data.get("final_val", {}).get("mae"),
            "train_rmse": data.get("final_train", {}).get("rmse"),
            "train_mae": data.get("final_train", {}).get("mae"),
        }
    )

out = runs_root / "temporal_80_20_l2_search.md"
lines = [
    "# Temporal 80/20 L2 Search",
    "",
    "| Run | L2 lambda | Best epoch | Val RMSE | Val MAE | Train RMSE | Train MAE |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['run']} | {row['l2_lambda']} | {row['best_epoch']} | "
        f"{row['val_rmse']:.6f} | {row['val_mae']:.6f} | "
        f"{row['train_rmse']:.6f} | {row['train_mae']:.6f} |"
    )
out.write_text("\n".join(lines) + "\n")
print(out)
PY
}

main() {
  cd "$ROOT"
  log "temporal 80/20 L2 search started"
  log "values: ${VALUES[*]}"

  local value
  for value in "${VALUES[@]}"; do
    local run_dir
    run_dir=$(ROOT="$ROOT" RUNS_ROOT="$RUNS_ROOT" bash "$ROOT/run_scripts/next_evimo_training_run_dir.sh")
    mkdir -p "$run_dir/logs"
    log "starting L2=$value in $run_dir"
    RUN_DIR="$run_dir" L2_LAMBDA="$value" WEIGHT_DECAY=0 bash "$ROOT/run_scripts/train_evimo_temporal_80_20_node2.sh" \
      2>&1 | tee "$run_dir/logs/l2_search_wrapper.log"
    log "finished L2=$value in $run_dir"
    summarize | tee -a "$LOG_DIR/l2_search.log"
  done

  log "temporal 80/20 L2 search done"
}

main "$@"
