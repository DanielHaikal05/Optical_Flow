#!/usr/bin/env bash
set -euo pipefail

batch_rel="${BATCH_REL:?BATCH_REL is required}"
python_bin="${PYTHON_BIN:?PYTHON_BIN is required}"
sample_count="${EVIMO_BOX_SWEEP_SAMPLE_COUNT:-30}"
sample_seed="${EVIMO_BOX_SWEEP_SAMPLE_SEED:-20260818}"
random_seed="${EVMOTIONSEG_RANDOM_SEED:-20260818}"
max_jobs="${EVIMO_BOX_SWEEP_MAX_JOBS:-3}"
smooth_values="${EVIMO_BOX_SWEEP_SMOOTH_VALUES:-1000,2500,5000,10000,25000,50000,100000}"
label_values="${EVIMO_BOX_SWEEP_LABEL_VALUES:-25000,75000,150000,300000,600000,1200000,2400000}"
sequences="${EVIMO_BOX_SWEEP_SEQUENCES:-seq_00 seq_02 seq_03 seq_04 seq_05 seq_06 seq_07 seq_08 seq_09 seq_10 seq_11}"

export PYTHONPATH="$(pwd)/VecKM_flow:${PYTHONPATH:-}"

build_dir="/tmp/evimo_box_remaining_param_sweep_build"
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}"

read -r -a sequence_array <<< "${sequences}"
printf "%s\n" "${sequence_array[@]}" > "${batch_rel}/sequences.txt"

active_jobs=0
failures=0
for seq in "${sequence_array[@]}"; do
  out_rel="${batch_rel}/${seq}"
  mkdir -p "${out_rel}/logs"
  (
    set -euo pipefail
    echo "Starting ${seq} at $(date -Is)"
    "${python_bin}" EvMotionSeg/tools/sweep_evimo_box_npz_scene_terms.py \
      --sequence-npz "Datasets/EVIMO/train/box/npz/${seq}.npz" \
      --output-dir "${out_rel}" \
      --binary "${build_dir}/motion_segmentation_standalone" \
      --cache-dir "VecKM_flow/outputs/evimo_box_${seq}_npz_cache" \
      --sample-count "${sample_count}" \
      --sample-seed "${sample_seed}" \
      --random-seed "${random_seed}" \
      --smooth-values "${smooth_values}" \
      --label-values "${label_values}" \
      --training-set EVIMO \
      --ensemble 3 \
      --no-auto-scale-time \
      --cleanup-run-dirs
    echo "Finished ${seq} at $(date -Is)"
  ) > "${out_rel}/logs/driver.log" 2>&1 &
  echo "$!" > "${out_rel}/job.pid"
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= max_jobs )); then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active_jobs=$((active_jobs - 1))
  fi
done

while (( active_jobs > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active_jobs=$((active_jobs - 1))
done

echo "Finished batch at $(date -Is) with ${failures} failed sequence job(s)."
exit "${failures}"
