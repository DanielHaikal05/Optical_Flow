#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}/VecKM_flow:${PYTHONPATH:-}"

python_bin="${PYTHON_BIN:-${repo_root}/E-MoFlow/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

sequence_dir="${DSEC_SEQUENCE_DIR:-Datasets/DSEC/zurich_city_00_a}"
sequence_label="${DSEC_SEQUENCE_LABEL:-$(basename "${sequence_dir}")}"
run_group="${ZURICH00A_SCENE_TERM_GROUP:-EvMotionSeg/data/dsec_zurich_city_00_a_scene_term_sweep_$(date +%Y%m%d_%H%M%S)}"
base_run="${ZURICH00A_SCENE_TERM_BASE_RUN:-${run_group}/fixed_veckm_text_base}"
cache_root="${ZURICH00A_SCENE_TERM_CACHE_ROOT:-${run_group}/veckm_cache}"
sample_dir="${cache_root}/${sequence_label}_density10k_p0p75"
veckm_out_root="${cache_root}/veckm"
veckm_pred_dir="${veckm_out_root}/$(basename "${sample_dir}")"
build_dir="${ZURICH00A_SCENE_TERM_BUILD_DIR:-/tmp/dsec_evmotionseg_zurich00a_scene_terms_build}"
result_dir="${run_group}/scene_term_results"
sample_count="${ZURICH00A_SCENE_TERM_SAMPLE_COUNT:-75}"
sample_seed="${ZURICH00A_SCENE_TERM_SAMPLE_SEED:-20260813}"
random_seed="${EVMOTIONSEG_RANDOM_SEED:-20260813}"
min_total_points="${ZURICH00A_SCENE_TERM_MIN_TOTAL_POINTS:-200}"
min_imo_points="${ZURICH00A_SCENE_TERM_MIN_IMO_POINTS:-20}"
smooth_base="${EVMOTIONSEG_SMOOTH_BASE:-7000}"
label_base="${EVMOTIONSEG_LABEL_BASE:-90000}"
cleanup_run_dirs="${EVMOTIONSEG_CLEANUP_RUN_DIRS:-1}"

mkdir -p "${run_group}/logs"
printf "%s\n" "${run_group}" > EvMotionSeg/data/dsec_zurich_city_00_a_scene_term_sweep_latest_path.txt
status_log="${run_group}/logs/status.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

need_base=0
for file in events.txt flow_xy.txt undistorted_normalized_xy.txt evmotionseg_input_summary.json; do
  if [[ ! -f "${base_run}/${file}" ]]; then
    need_base=1
  fi
done

log "Run group: ${run_group}"
log "Sequence: ${sequence_dir}"
log "Fixed VecKM EvMotionSeg base: ${base_run}"
log "Sweep sample_count=${sample_count}, smooth_base=${smooth_base}, label_base=${label_base}, seed=${random_seed}"

has_auto_mos=0
mos_roots=()
if [[ -n "${DSEC_MOS_ROOT:-}" ]]; then
  mos_roots+=("${DSEC_MOS_ROOT}")
fi
mos_roots+=(
  "$(dirname "${sequence_dir}")/DSEC_MOS"
  "$(dirname "${sequence_dir}")/DSEC-MOS"
  "$(dirname "$(dirname "${sequence_dir}")")/DSEC_MOS"
  "$(dirname "$(dirname "${sequence_dir}")")/DSEC-MOS"
)
for root in "${mos_roots[@]}"; do
  for split in gt_mask training testing ""; do
    if [[ "${split}" == "gt_mask" ]]; then
      candidate="${root}/gt_mask/${sequence_label}"
    elif [[ -n "${split}" ]]; then
      candidate="${root}/${split}/${sequence_label}/gt_mask"
    else
      candidate="${root}/${sequence_label}/gt_mask"
    fi
    if [[ -d "${candidate}" ]]; then
      has_auto_mos=1
    fi
  done
done
for candidate in \
  "${sequence_dir}/gt_mask" \
  "${sequence_dir}/mos/gt_mask" \
  "${sequence_dir}/dsec_mos/gt_mask" \
  "${sequence_dir}/motion_segmentation/gt_mask" \
  "${sequence_dir}/moving_object_segmentation/gt_mask" \
  "${sequence_dir}/mos" \
  "${sequence_dir}/dsec_mos" \
  "${sequence_dir}/motion_segmentation" \
  "${sequence_dir}/moving_object_segmentation"; do
  if [[ -d "${candidate}" ]]; then
    has_auto_mos=1
  fi
done
if [[ -z "${DSEC_MOS_GT_DIR:-}" && "${has_auto_mos}" == "0" && "${ALLOW_DSEC_SEMANTIC_PROXY:-0}" != "1" ]]; then
  log "Missing DSEC-MOS GT masks. Set DSEC_MOS_GT_DIR, place masks under a known MOS directory, or set ALLOW_DSEC_SEMANTIC_PROXY=1 for a non-MOS semantic-car proxy."
  exit 1
fi
if [[ -n "${DSEC_MOS_GT_DIR:-}" && ! -d "${DSEC_MOS_GT_DIR}" ]]; then
  log "DSEC_MOS_GT_DIR does not exist or is not a directory: ${DSEC_MOS_GT_DIR}"
  exit 1
fi

log "Building deterministic EvMotionSeg standalone binary."
EvMotionSeg/tools/build_standalone_portable.sh "${build_dir}" >"${run_group}/logs/build.log" 2>&1
binary="${build_dir}/motion_segmentation_standalone"

if (( need_base )); then
  log "Fixed NF text base is missing; preparing DSEC events and VecKM cache once."
  mkdir -p "${cache_root}"
  "${python_bin}" EvMotionSeg/tools/prepare_dsec_veckm_input.py \
    --sequence-dir "${sequence_dir}" \
    --output-dir "${sample_dir}" \
    --interval-s 0.1 \
    --sampling-mode density \
    --max-events-per-interval 10000 \
    --grid-cols 8 \
    --grid-rows 6 \
    --density-power 0.75 \
    --density-offset 1.0 \
    --preview-stride 1 \
    --overwrite \
    >"${run_group}/logs/prepare_dsec_veckm_input.log" 2>&1

  log "Running VecKM once for the fixed NF cache."
  "${python_bin}" VecKM_flow/run_veckm_evimo2_sliding.py \
    --data-root "$(dirname "${sample_dir}")" \
    --sequences "$(basename "${sample_dir}")" \
    --output-dir "${veckm_out_root}" \
    --training-set DSEC \
    --ensemble 3 \
    --save-undistorted \
    >"${run_group}/logs/veckm.log" 2>&1

  log "Converting fixed VecKM normal flow to EvMotionSeg text input."
  "${python_bin}" EvMotionSeg/tools/prepare_drone_sequence.py text \
    "${sample_dir}" \
    "${veckm_pred_dir}" \
    "${base_run}" \
    --flow-file dataset_pred_flow_DSEC.npy \
    --timestamp-scale 1e-6 \
    --chunk-size 250000 \
    >"${run_group}/logs/evmotionseg_text.log" 2>&1
  rm -rf "${base_run}/event_preview"
  cp -a "${sample_dir}/event_preview" "${base_run}/event_preview"
else
  log "Reusing existing fixed NF text base."
fi

gt_args=(--gt-mode auto)
if [[ -n "${DSEC_MOS_GT_DIR:-}" ]]; then
  gt_args=(--gt-mode mask --gt-mask-dir "${DSEC_MOS_GT_DIR}")
  if [[ -n "${DSEC_MOS_GT_TIMESTAMPS:-}" ]]; then
    gt_args+=(--gt-timestamps "${DSEC_MOS_GT_TIMESTAMPS}")
  fi
elif [[ -n "${DSEC_MOS_ROOT:-}" ]]; then
  for split in gt_mask training testing ""; do
    if [[ "${split}" == "gt_mask" ]]; then
      candidate="${DSEC_MOS_ROOT}/gt_mask/${sequence_label}"
    elif [[ -n "${split}" ]]; then
      candidate="${DSEC_MOS_ROOT}/${split}/${sequence_label}/gt_mask"
    else
      candidate="${DSEC_MOS_ROOT}/${sequence_label}/gt_mask"
    fi
    if [[ -d "${candidate}" ]]; then
      gt_args=(--gt-mode mask --gt-mask-dir "${candidate}")
      break
    fi
  done
elif [[ "${ALLOW_DSEC_SEMANTIC_PROXY:-0}" == "1" ]]; then
  gt_args=(--gt-mode semantic --allow-semantic-proxy)
fi

log "Starting scene-dependent 7x7 smooth/label sweep."
sweep_extra_args=()
if [[ "${cleanup_run_dirs}" == "1" ]]; then
  sweep_extra_args+=(--cleanup-run-dirs)
fi
"${python_bin}" EvMotionSeg/tools/sweep_dsec_evmotionseg_scene_terms.py \
  --sequence-dir "${sequence_dir}" \
  --base-run "${base_run}" \
  --output-dir "${result_dir}" \
  --binary "${binary}" \
  --sample-count "${sample_count}" \
  --sample-seed "${sample_seed}" \
  --random-seed "${random_seed}" \
  --min-total-points "${min_total_points}" \
  --min-imo-points "${min_imo_points}" \
  --smooth-base "${smooth_base}" \
  --label-base "${label_base}" \
  --data-term 1 \
  --downsample-rate 1 \
  --graph-cut-iterations 10 \
  --motion-seg-iterations 4 \
  --max-labels 64 \
  --imo-background-mode background_fit \
  --background-label-error-ratio 3.0 \
  --background-label-min-fraction 0.10 \
  --smoothness-mode constant \
  --smoothness-flow-sigma 25 \
  --smoothness-min-weight 0.05 \
  --candidate-angle-eps 45 \
  --candidate-length-ratio-eps 2.0 \
  --initial-candidate-count 12 \
  --tracked-candidate-bonus 6 \
  --label-track-min-fraction 0.02 \
  --label-track-max-fraction 0.20 \
  --label-retention-mode legacy \
  "${gt_args[@]}" \
  "${sweep_extra_args[@]}" \
  >"${run_group}/logs/scene_term_sweep.log" 2>&1

log "Scene-dependent sweep complete: ${result_dir}"
