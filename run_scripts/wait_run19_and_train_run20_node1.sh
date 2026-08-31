#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
SWEEP_ROOT=${SWEEP_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps}
RUN19_DIR=${RUN19_DIR:-$SWEEP_ROOT/training_runs/19}
RUN20_DIR=${RUN20_DIR:-$SWEEP_ROOT/training_runs/20}
LOG_DIR="$RUN20_DIR/logs"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/run20_queue.log"
}

run19_active() {
  if [ -f "$RUN19_DIR/logs/run19.pid" ]; then
    local pid
    pid=$(cat "$RUN19_DIR/logs/run19.pid")
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  pgrep -f "train_evimo_multi_sequence_param_f1_predictor.py .*training_runs/19" >/dev/null 2>&1
}

main() {
  cd "$ROOT"
  log "queue started for run 20"
  log "waiting for run 19: $RUN19_DIR"

  while [ ! -f "$RUN19_DIR/summary.json" ]; do
    if run19_active; then
      sleep 60
    else
      log "run 19 is not active and has no summary.json; refusing to start run 20"
      exit 1
    fi
  done

  if [ -f "$RUN20_DIR/summary.json" ]; then
    log "run 20 already has summary.json; nothing to do"
    exit 0
  fi

  if pgrep -f "train_evimo_multi_sequence_param_f1_predictor.py .*training_runs/20" >/dev/null 2>&1; then
    log "run 20 already appears active; nothing to do"
    exit 0
  fi

  log "run 19 complete; starting run 20 with wide64 encoder, 48-32-16 MLP, and LR scheduler"
  RUN_DIR="$RUN20_DIR" \
    L2_LAMBDA=1e-5 \
    WEIGHT_DECAY=0 \
    DROPOUT=0.2 \
    ENCODER_VARIANT=wide64 \
    MLP_HIDDEN_DIMS=48,32,16 \
    LR_SCHEDULER=reduce_on_plateau \
    LR_SCHEDULER_PATIENCE=5 \
    LR_SCHEDULER_FACTOR=0.5 \
    MIN_LR=0 \
    bash "$ROOT/run_scripts/train_evimo_temporal_80_20_node2.sh" \
    2>&1 | tee "$LOG_DIR/run20_wrapper.log"
  log "run 20 finished"
}

main "$@"
