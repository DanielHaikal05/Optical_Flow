#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
RUN17_DIR=${RUN17_DIR:-$SWEEP_ROOT/training_runs/17}
RUN18_DIR=${RUN18_DIR:-$SWEEP_ROOT/training_runs/18}
LOG_DIR="$RUN18_DIR/logs"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/run18_queue.log"
}

run17_active() {
  if [ -f "$RUN17_DIR/logs/run17.pid" ]; then
    local pid
    pid=$(cat "$RUN17_DIR/logs/run17.pid")
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  pgrep -f "train_evimo_multi_sequence_param_f1_predictor.py .*training_runs/17" >/dev/null 2>&1
}

main() {
  cd "$ROOT"
  log "queue started for run 18"
  log "waiting for run 17: $RUN17_DIR"

  while [ ! -f "$RUN17_DIR/summary.json" ]; do
    if run17_active; then
      sleep 60
    else
      log "run 17 is not active and has no summary.json; refusing to start run 18"
      exit 1
    fi
  done

  if [ -f "$RUN18_DIR/summary.json" ]; then
    log "run 18 already has summary.json; nothing to do"
    exit 0
  fi

  if pgrep -f "train_evimo_multi_sequence_param_f1_predictor.py .*training_runs/18" >/dev/null 2>&1; then
    log "run 18 already appears active; nothing to do"
    exit 0
  fi

  log "run 17 complete; starting run 18 with wide64 encoder and LR scheduler"
  RUN_DIR="$RUN18_DIR" \
    L2_LAMBDA=1e-5 \
    WEIGHT_DECAY=0 \
    DROPOUT=0.2 \
    ENCODER_VARIANT=wide64 \
    LR_SCHEDULER=reduce_on_plateau \
    LR_SCHEDULER_PATIENCE=5 \
    LR_SCHEDULER_FACTOR=0.5 \
    MIN_LR=0 \
    bash "$ROOT/run_scripts/train_evimo_temporal_80_20_node2.sh" \
    2>&1 | tee "$LOG_DIR/run18_wrapper.log"
  log "run 18 finished"
}

main "$@"
