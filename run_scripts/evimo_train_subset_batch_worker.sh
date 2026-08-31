#!/usr/bin/env bash
set -euo pipefail

subset="${EVIMO_TRAIN_SUBSET:?EVIMO_TRAIN_SUBSET is required, e.g. floor or wall}"
batch_rel="${BATCH_REL:?BATCH_REL is required}"
python_bin="${PYTHON_BIN:?PYTHON_BIN is required}"
sample_count="${EVIMO_SWEEP_SAMPLE_COUNT:-30}"
sample_seed="${EVIMO_SWEEP_SAMPLE_SEED:-20260818}"
random_seed="${EVMOTIONSEG_RANDOM_SEED:-20260818}"
max_jobs="${EVIMO_SWEEP_MAX_JOBS:-1}"
smooth_values="${EVIMO_SWEEP_SMOOTH_VALUES:-1000,2500,5000,10000,25000,50000,100000}"
label_values="${EVIMO_SWEEP_LABEL_VALUES:-25000,75000,150000,300000,600000,1200000,2400000}"
sequences="${EVIMO_SWEEP_SEQUENCES:-seq_00 seq_01 seq_02}"
build_dir="${EVMOTIONSEG_BUILD_DIR:-/tmp/evimo_train_param_sweep_build}"

export PYTHONPATH="$(pwd)/VecKM_flow:${PYTHONPATH:-}"

EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}"

read -r -a sequence_array <<< "${sequences}"
mkdir -p "${batch_rel}/logs"
printf "%s\n" "${sequence_array[@]}" > "${batch_rel}/sequences.txt"
printf "%s\n" "${subset}" > "${batch_rel}/subset.txt"

active_jobs=0
failures=0
for seq in "${sequence_array[@]}"; do
  out_rel="${batch_rel}/${seq}"
  mkdir -p "${out_rel}/logs"
  (
    set -euo pipefail
    echo "Starting ${subset}/${seq} at $(date -Is)"
    "${python_bin}" EvMotionSeg/tools/sweep_evimo_box_npz_scene_terms.py \
      --sequence-npz "Datasets/EVIMO/train/${subset}/npz/${seq}.npz" \
      --output-dir "${out_rel}" \
      --binary "${build_dir}/motion_segmentation_standalone" \
      --cache-dir "VecKM_flow/outputs/evimo_${subset}_${seq}_npz_cache" \
      --sample-count "${sample_count}" \
      --sample-seed "${sample_seed}" \
      --random-seed "${random_seed}" \
      --smooth-values "${smooth_values}" \
      --label-values "${label_values}" \
      --training-set EVIMO \
      --ensemble 3 \
      --no-auto-scale-time \
      --cleanup-run-dirs
    echo "Finished ${subset}/${seq} at $(date -Is)"
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

echo "Finished ${subset} batch at $(date -Is) with ${failures} failed sequence job(s)." | tee "${batch_rel}/logs/batch.done"
exit "${failures}"
