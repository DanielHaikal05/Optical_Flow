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
sequence_rel="Datasets/EVIMO/train/box/npz/seq_01.npz"
latest_rel="EvMotionSeg/data/evimo_box_seq01_param_sweep_latest_path.txt"
smooth_values="${EVIMO_BOX_SWEEP_SMOOTH_VALUES:-1000,2500,5000,10000,25000,50000,100000}"
label_values="${EVIMO_BOX_SWEEP_LABEL_VALUES:-25000,75000,150000,300000,600000,1200000,2400000}"

sync_code() {
  rsync -av \
    EvMotionSeg/tools/sweep_evimo_box_npz_scene_terms.py \
    EvMotionSeg/tools/train_nf_param_f1_predictor.py \
    EvMotionSeg/tools/build_standalone_portable.sh \
    "${node2}:${remote_root}/EvMotionSeg/tools/"
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
  rsync -av --partial --progress "${sequence_rel}" "${node2}:${remote_root}/${sequence_rel}"
}

remote_latest_path() {
  ssh "${node2}" "cd '${remote_root}' && test -f '${latest_rel}' && cat '${latest_rel}'"
}

case "${cmd}" in
  start)
    timestamp="$(date +%Y%m%d_%H%M%S)"
    out_rel="EvMotionSeg/data/evimo_box_seq01_param_sweep_${timestamp}"
    echo "Syncing code to ${node2}:${remote_root}"
    sync_code
    echo "Syncing ${sequence_rel} to node 2"
    sync_data
    echo "Starting remote sweep: ${out_rel}"
    ssh "${node2}" "cd '${remote_root}' && mkdir -p '${out_rel}/logs' && printf '%s\n' '${out_rel}' > '${latest_rel}' && (nohup bash -lc '
set -euo pipefail
cd \"${remote_root}\"
export PYTHONPATH=\"${remote_root}/VecKM_flow:\${PYTHONPATH:-}\"
build_dir=\"/tmp/evimo_box_seq01_param_sweep_build\"
EvMotionSeg/tools/build_standalone_portable.sh \"\${build_dir}\"
\"${python_bin}\" EvMotionSeg/tools/sweep_evimo_box_npz_scene_terms.py \
  --sequence-npz \"${sequence_rel}\" \
  --output-dir \"${out_rel}\" \
  --binary \"\${build_dir}/motion_segmentation_standalone\" \
  --cache-dir \"VecKM_flow/outputs/evimo_box_seq01_npz_cache\" \
  --sample-count \"${sample_count}\" \
  --sample-seed \"${sample_seed}\" \
  --random-seed \"${random_seed}\" \
  --smooth-values \"${smooth_values}\" \
  --label-values \"${label_values}\" \
  --training-set EVIMO \
  --ensemble 3 \
  --no-auto-scale-time \
  --cleanup-run-dirs
' > '${out_rel}/logs/driver.log' 2>&1 & echo \$! > '${out_rel}/job.pid')"
    echo "Remote output: ${out_rel}"
    echo "Status: $0 status"
    echo "Tail:   $0 tail"
    echo "Pull:   $0 pull"
    ;;
  status)
    out_rel="$(remote_latest_path)"
    ssh "${node2}" "cd '${remote_root}' && echo 'output: ${out_rel}' && if test -f '${out_rel}/job.pid'; then pid=\$(cat '${out_rel}/job.pid'); echo \"pid: \$pid\"; ps -p \"\$pid\" -o pid,etime,stat,cmd || true; else pgrep -af 'sweep_evimo_box_npz_scene_terms.py' || true; fi && find '${out_rel}' -maxdepth 1 -type f -printf '%p %s bytes\n' 2>/dev/null | sort && tail -40 '${out_rel}/logs/driver.log' 2>/dev/null || true"
    ;;
  tail)
    out_rel="$(remote_latest_path)"
    ssh -t "${node2}" "cd '${remote_root}' && tail -f '${out_rel}/logs/driver.log'"
    ;;
  pull)
    out_rel="$(remote_latest_path)"
    mkdir -p "${out_rel}"
    rsync -av --partial "${node2}:${remote_root}/${out_rel}/" "${out_rel}/"
    echo "Pulled to ${repo_root}/${out_rel}"
    ;;
  *)
    echo "Usage: $0 [start|status|tail|pull]" >&2
    exit 2
    ;;
esac
