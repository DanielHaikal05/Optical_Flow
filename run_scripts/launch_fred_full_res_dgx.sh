#!/usr/bin/env bash
set -euo pipefail

cd /home/DanielH/Optical_Flow

run_root="VecKM_flow/outputs/fred_full_res_waited_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_root}/logs"
printf '%s\n' "${run_root}" > VecKM_flow/outputs/fred_full_res_waited_latest_path.txt
ln -sfn "$(basename "${run_root}")" VecKM_flow/outputs/fred_full_res_waited_latest
status_log="${run_root}/logs/status.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${status_log}"
}

log "Full-resolution FRED job staged on $(hostname)."
log "Waiting for existing training process to finish before using the GPU."

while pgrep -u "${USER}" -af 'python[0-9.]*[[:space:]].*train.py|python[[:space:]]+train.py' >/dev/null; do
  pgrep -u "${USER}" -af 'python[0-9.]*[[:space:]].*train.py|python[[:space:]]+train.py' | tee -a "${status_log}" >/dev/null || true
  sleep 300
done

log "No train.py process detected; starting full-resolution pipeline."

EvMotionSeg/tools/build_standalone.sh /tmp/fred_evmotionseg_build >>"${run_root}/logs/build_standalone.log" 2>&1

for seq in 1 2; do
  log "Starting FRED ${seq} VecKM full-event inference."
  vec_out="VecKM_flow/outputs/fred_${seq}_full_veckm_e1_us"
  ev_out="EvMotionSeg/data/fred_${seq}_full"
  mkdir -p "${vec_out}/logs" "${ev_out}"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/home/DanielH/Optical_Flow \
  /home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python \
    VecKM_flow/run_veckm_evimo2_sliding.py \
    --data-root "Datasets/FRED/${seq}" \
    --sequences event_stream_full \
    --output-dir "${vec_out}" \
    --training-set EVIMO \
    --ensemble 1 \
    --timestamp-scale 1e-6 \
    --save-undistorted \
    >"${vec_out}/logs/run.log" 2>&1

  log "Converting FRED ${seq} full-event VecKM output for EvMotionSeg."
  /home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python \
    EvMotionSeg/tools/prepare_drone_sequence.py text \
    "Datasets/FRED/${seq}/event_stream_full" \
    "${vec_out}/event_stream_full" \
    "${ev_out}" \
    --flow-file dataset_pred_flow_EVIMO.npy \
    --timestamp-scale 1e-6 \
    >"${ev_out}/prepare_text.log" 2>&1

  find "${ev_out}" -maxdepth 1 -type f \( -name 'timestamp.csv' -o -name 'motion_labels.csv' \) -delete
  find "${ev_out}/results" "${ev_out}/results_next" "${ev_out}/results_imo" \
    -maxdepth 1 -type f -name '*.png' -delete 2>/dev/null || true

  intervals=0
  case "${seq}" in
    1) intervals=3516 ;;
    2) intervals=3530 ;;
  esac

  log "Running EvMotionSeg full-event IMO segmentation for FRED ${seq}."
  /tmp/fred_evmotionseg_build/motion_segmentation_standalone \
    --data_file_path "${ev_out}" \
    --interval 0.033333 \
    --width 1280 \
    --height 720 \
    --downsample_rate 1 \
    --fx 1280 \
    --fy 1280 \
    --data_term 1 \
    --smooth_term 6000 \
    --label_term 60000 \
    --GraphCutIteration 10 \
    --MotionSegIteration 4 \
    --max_labels 32 \
    --num_intervals "${intervals}" \
    --imo_background_mode all_remaining \
    >"${ev_out}/run_imo.log" 2>&1

  log "Evaluating FRED ${seq} full-event results against GT-only YOLO labels from frame 700."
  prediction_frame_offset=0
  case "${seq}" in
    1) prediction_frame_offset=85 ;;
    2) prediction_frame_offset=71 ;;
  esac
  /home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python \
    tools/evaluate_fred_evmotionseg_yolo.py \
    --results-dir "${ev_out}/results_imo" \
    --event-yolo-dir "Datasets/FRED/${seq}/Event_YOLO" \
    --output-dir "${ev_out}/evaluation_yolo_imo_from700_gt_only_precision" \
    --width 1280 \
    --height 720 \
    --min-component-area 2 \
    --white-threshold 250 \
    --iou-threshold 0.01 \
    --coverage-threshold 0.05 \
    --start-frame 700 \
    --prediction-frame-offset "${prediction_frame_offset}" \
    --gt-only \
    >"${ev_out}/evaluation.log" 2>&1

  /home/DanielH/Optical_Flow/E-MoFlow/.venv/bin/python \
    tools/evaluate_fred_evmotionseg_yolo.py \
    --results-dir "${ev_out}/results" \
    --event-yolo-dir "Datasets/FRED/${seq}/Event_YOLO" \
    --output-dir "${ev_out}/evaluation_yolo_all_from700_gt_only_precision" \
    --width 1280 \
    --height 720 \
    --min-component-area 2 \
    --white-threshold 250 \
    --iou-threshold 0.01 \
    --coverage-threshold 0.05 \
    --start-frame 700 \
    --prediction-frame-offset "${prediction_frame_offset}" \
    --gt-only \
    >"${ev_out}/evaluation_all.log" 2>&1

  log "Completed FRED ${seq} full-resolution run."
done

log "Full-resolution FRED job complete."
