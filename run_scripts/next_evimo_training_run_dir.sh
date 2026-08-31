#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/DanielH/Optical_Flow}
RUNS_ROOT=${RUNS_ROOT:-$ROOT/EvMotionSeg/data/evimo_sweeps/training_runs}

mkdir -p "$RUNS_ROOT"

next=1
while [ -e "$RUNS_ROOT/$next" ]; do
  next=$((next + 1))
done

mkdir -p "$RUNS_ROOT/$next"
printf '%s\n' "$RUNS_ROOT/$next"
