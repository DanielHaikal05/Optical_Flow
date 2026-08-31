#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

cmd="${1:-start}"
node2="${DGX_NODE2:-DanielH@129.97.250.113}"
remote_root="${DGX_REMOTE_ROOT:-/home/DanielH/Optical_Flow}"
python_bin="${PYTHON_BIN:-/home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python}"
sample_count="${EVIMO_BOX_SWEEP_SAMPLE_COUNT:-30}"
sample_seed="${EVIMO_BOX_SWEEP_SAMPLE_SEED:-20260818}"
random_seed="${EVMOTIONSEG_RANDOM_SEED:-20260818}"
max_jobs="${EVIMO_BOX_SWEEP_MAX_JOBS:-3}"
smooth_values="${EVIMO_BOX_SWEEP_SMOOTH_VALUES:-1000,2500,5000,10000,25000,50000,100000}"
label_values="${EVIMO_BOX_SWEEP_LABEL_VALUES:-25000,75000,150000,300000,600000,1200000,2400000}"
sequences="${EVIMO_BOX_SWEEP_SEQUENCES:-seq_00 seq_02 seq_03 seq_04 seq_05 seq_06 seq_07 seq_08 seq_09 seq_10 seq_11}"
batch_latest_rel="EvMotionSeg/data/evimo_box_remaining_param_sweeps_latest_path.txt"

sync_code() {
  rsync -av \
    EvMotionSeg/tools/sweep_evimo_box_npz_scene_terms.py \
    EvMotionSeg/tools/train_nf_param_f1_predictor.py \
    EvMotionSeg/tools/build_standalone_portable.sh \
    "${node2}:${remote_root}/EvMotionSeg/tools/"
  rsync -av \
    run_scripts/evimo_box_remaining_batch_worker.sh \
    "${node2}:${remote_root}/run_scripts/"
  rsync -av \
    EvMotionSeg/src/MotionSeg.cpp \
    EvMotionSeg/src/MotionSeg_standalone.cpp \
    "${node2}:${remote_root}/EvMotionSeg/src/"
  rsync -av \
    EvMotionSeg/include/motionseg/MotionSeg.h \
    "${node2}:${remote_root}/EvMotionSeg/include/motionseg/"
}

sync_data() {
  ssh "${node2}" "mkdir -p '${remote_root}/Datasets/EVIMO/train/box/npz'"
  for seq in ${sequences}; do
    local path="Datasets/EVIMO/train/box/npz/${seq}.npz"
    echo "Syncing ${path}"
    rsync -av --partial --progress "${path}" "${node2}:${remote_root}/${path}"
  done
}

remote_batch_path() {
  ssh "${node2}" "cd '${remote_root}' && test -f '${batch_latest_rel}' && cat '${batch_latest_rel}'"
}

case "${cmd}" in
  start)
    timestamp="$(date +%Y%m%d_%H%M%S)"
    batch_rel="EvMotionSeg/data/evimo_box_remaining_param_sweeps_${timestamp}"
    echo "Syncing code to ${node2}:${remote_root}"
    sync_code
    sync_data
    echo "Starting remote batch: ${batch_rel}"
    ssh "${node2}" "cd '${remote_root}' && mkdir -p '${batch_rel}/logs' && printf '%s\n' '${batch_rel}' > '${batch_latest_rel}' && chmod +x run_scripts/evimo_box_remaining_batch_worker.sh && (BATCH_REL='${batch_rel}' PYTHON_BIN='${python_bin}' EVIMO_BOX_SWEEP_SAMPLE_COUNT='${sample_count}' EVIMO_BOX_SWEEP_SAMPLE_SEED='${sample_seed}' EVMOTIONSEG_RANDOM_SEED='${random_seed}' EVIMO_BOX_SWEEP_MAX_JOBS='${max_jobs}' EVIMO_BOX_SWEEP_SMOOTH_VALUES='${smooth_values}' EVIMO_BOX_SWEEP_LABEL_VALUES='${label_values}' EVIMO_BOX_SWEEP_SEQUENCES='${sequences}' nohup run_scripts/evimo_box_remaining_batch_worker.sh > '${batch_rel}/logs/batch.log' 2>&1 & echo \$! > '${batch_rel}/batch.pid')"
    echo "Remote batch output: ${batch_rel}"
    echo "Status: $0 status"
    echo "Pull:   $0 pull"
    ;;
  status)
    batch_rel="$(remote_batch_path)"
    ssh "${node2}" "cd '${remote_root}' && echo 'batch: ${batch_rel}' && if test -f '${batch_rel}/batch.pid'; then pid=\$(cat '${batch_rel}/batch.pid'); echo \"batch pid: \$pid\"; ps -p \"\$pid\" -o pid,etime,stat,cmd || true; fi && echo 'files:' && find '${batch_rel}' -maxdepth 2 -type f \\( -name summary.json -o -name f1_matrix.md -o -name job.pid -o -name grid_metrics.csv \\) -printf '%p %s bytes\n' 2>/dev/null | sort && echo 'running sweeps:' && pgrep -af 'sweep_evimo_box_npz_scene_terms.py' || true && echo 'tails:' && for seq in \$(cat '${batch_rel}/sequences.txt' 2>/dev/null); do echo \"--- \$seq ---\"; tail -12 '${batch_rel}/'\$seq'/logs/driver.log' 2>/dev/null || true; done && echo 'batch log:' && tail -20 '${batch_rel}/logs/batch.log' 2>/dev/null || true"
    ;;
  pull)
    batch_rel="$(remote_batch_path)"
    mkdir -p "${batch_rel}"
    rsync -av --partial "${node2}:${remote_root}/${batch_rel}/" "${batch_rel}/"
    echo "Pulled to ${repo_root}/${batch_rel}"
    ;;
  *)
    echo "Usage: $0 [start|status|pull]" >&2
    exit 2
    ;;
esac
