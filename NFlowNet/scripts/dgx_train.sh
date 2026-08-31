#!/usr/bin/env bash
set -euo pipefail

ACTION="start"
NODE_SELECTOR="${DGX_NODE:-1}"
REMOTE_CODE_DIR="${DGX_REMOTE_CODE_DIR:-Optical_Flow/NFlowNet}"
REMOTE_DATA_DIR="${DGX_REMOTE_DATA_DIR:-Optical_Flow/Datasets/TartanAir}"
REMOTE_ROOT="${DGX_REMOTE_ROOT:-$REMOTE_DATA_DIR}"
ROOT_EXPLICIT=0
if [ -n "${DGX_REMOTE_ROOT:-}" ]; then
  ROOT_EXPLICIT=1
fi
REMOTE_VENV="${DGX_REMOTE_VENV:-.venv}"
REMOTE_LOG_DIR="${DGX_REMOTE_LOG_DIR:-logs/dgx}"
LOCAL_DATA_DIR="${DGX_LOCAL_DATA_DIR:-Datasets/TartanAir}"
TRAIN_CMD="${DGX_TRAIN_CMD:-python train.py}"
TRAIN_VARIANT="${DGX_TRAIN_VARIANT:-train}"
FOLLOW_LOGS=1
SYNC_CODE=1
SYNC_DATA=0
DRY_RUN=0
ACCESS_FILE="${DGX_ACCESS_FILE:-$HOME/dgx_access}"
AUTO_PASSWORD="${DGX_AUTO_PASSWORD:-1}"
PASSWORD_FILE=""

usage() {
  cat <<USAGE
Usage:
  NFlowNet/scripts/dgx_train.sh [start|status|tail|tail-once|stop|sync-code|sync-data] [options]

Default behavior:
  start: sync NFlowNet code to DGX node 1, then run "python train.py" there.

Options:
  --node 1|2|all       DGX node target. Default: ${NODE_SELECTOR}
  --train-cmd CMD      Remote command to run from the NFlowNet directory.
                       Default: ${TRAIN_CMD}
  --raft               Use train_raft.py by exporting it remotely as train.py.
  --train              Use train.py. This is the default.
  --code-dir PATH      Remote NFlowNet path, relative to \$HOME unless absolute.
                       Default: ${REMOTE_CODE_DIR}
  --data-dir PATH      Remote TartanAir path, relative to \$HOME unless absolute.
                       Default: ${REMOTE_DATA_DIR}
  --root PATH          Value written to ROOT in the remote train.py before start.
                       Relative paths are resolved under remote \$HOME.
                       Default: ${REMOTE_ROOT}
  --local-data PATH    Local TartanAir path relative to workspace root, or absolute.
                       Default: ${LOCAL_DATA_DIR}
  --venv PATH          Remote virtualenv path, relative to --code-dir unless absolute.
                       Default: ${REMOTE_VENV}
  --log-dir PATH       Remote log path, relative to --code-dir unless absolute.
                       Default: ${REMOTE_LOG_DIR}
  --with-data          Sync TartanAir before starting.
  --no-sync-code       Do not sync NFlowNet before starting.
  --no-follow          For tail, print recent log lines and exit.
  --dry-run            Print actions without changing remote machines.
  -h, --help           Show this help.

Environment overrides:
  DGX_NODE1, DGX_NODE2, DGX_ACCESS_FILE, DGX_AUTO_PASSWORD, DGX_TRAIN_CMD,
  DGX_REMOTE_CODE_DIR, DGX_REMOTE_DATA_DIR, DGX_REMOTE_VENV, DGX_REMOTE_LOG_DIR,
  DGX_LOCAL_DATA_DIR, DGX_REMOTE_ROOT, DGX_TRAIN_VARIANT

Examples:
  NFlowNet/scripts/dgx_train.sh sync-data --node all
  NFlowNet/scripts/dgx_train.sh sync-code
  NFlowNet/scripts/dgx_train.sh
  NFlowNet/scripts/dgx_train.sh --raft
  NFlowNet/scripts/dgx_train.sh tail
  NFlowNet/scripts/dgx_train.sh tail-once
  DGX_TRAIN_CMD="python train.py --epochs 50" NFlowNet/scripts/dgx_train.sh
USAGE
}

cleanup_password_file() {
  if [ -n "$PASSWORD_FILE" ] && [ -f "$PASSWORD_FILE" ]; then
    rm -f "$PASSWORD_FILE"
  fi
}
trap cleanup_password_file EXIT

if [[ "${1:-}" =~ ^(start|status|tail|tail-once|stop|sync-code|sync-data)$ ]]; then
  ACTION="$1"
  if [ "$ACTION" = "tail-once" ]; then
    ACTION="tail"
    FOLLOW_LOGS=0
  fi
  shift
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --node)
      NODE_SELECTOR="$2"
      shift 2
      ;;
    --train-cmd)
      TRAIN_CMD="$2"
      shift 2
      ;;
    --raft)
      TRAIN_VARIANT="raft"
      shift
      ;;
    --train)
      TRAIN_VARIANT="train"
      shift
      ;;
    --code-dir)
      REMOTE_CODE_DIR="$2"
      shift 2
      ;;
    --data-dir)
      REMOTE_DATA_DIR="$2"
      shift 2
      ;;
    --root)
      REMOTE_ROOT="$2"
      ROOT_EXPLICIT=1
      shift 2
      ;;
    --local-data)
      LOCAL_DATA_DIR="$2"
      shift 2
      ;;
    --venv)
      REMOTE_VENV="$2"
      shift 2
      ;;
    --log-dir)
      REMOTE_LOG_DIR="$2"
      shift 2
      ;;
    --with-data)
      SYNC_DATA=1
      shift
      ;;
    --no-sync-code)
      SYNC_CODE=0
      shift
      ;;
    --no-follow)
      FOLLOW_LOGS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$ROOT_EXPLICIT" = "0" ]; then
  REMOTE_ROOT="$REMOTE_DATA_DIR"
fi

case "$TRAIN_VARIANT" in
  train|raft)
    ;;
  *)
    echo "DGX_TRAIN_VARIANT must be 'train' or 'raft'" >&2
    exit 2
    ;;
esac

TRAIN_CMD_B64="$(printf '%s' "$TRAIN_CMD" | base64 | tr -d '\n')"

workspace_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

nflownet_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

abs_or_workspace_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$(workspace_root)" "$1"
  fi
}

load_nodes() {
  local node1="${DGX_NODE1:-}"
  local node2="${DGX_NODE2:-}"

  if [ -z "$node1" ] || [ -z "$node2" ]; then
    if [ -f "$ACCESS_FILE" ]; then
      mapfile -t access_nodes < <(awk '/^ssh[[:space:]]+/ {print $2}' "$ACCESS_FILE")
      node1="${node1:-${access_nodes[0]:-}}"
      node2="${node2:-${access_nodes[1]:-}}"
    fi
  fi

  node1="${node1:-DanielH@129.97.250.112}"
  node2="${node2:-DanielH@129.97.250.113}"

  case "$NODE_SELECTOR" in
    1)
      printf '%s\n' "$node1"
      ;;
    2)
      printf '%s\n' "$node2"
      ;;
    all)
      printf '%s\n%s\n' "$node1" "$node2"
      ;;
    *)
      echo "--node must be 1, 2, or all" >&2
      exit 2
      ;;
  esac
}

load_password() {
  if [ ! -f "$ACCESS_FILE" ]; then
    return
  fi
  awk '
    BEGIN { want = 0 }
    tolower($0) ~ /^password:?[[:space:]]*$/ { want = 1; next }
    want && NF {
      print
      exit
    }
  ' "$ACCESS_FILE"
}

init_password_auth() {
  if [ "$AUTO_PASSWORD" != "1" ]; then
    return
  fi

  local password
  password="$(load_password || true)"
  if [ -z "$password" ]; then
    return
  fi

  if ! command -v sshpass >/dev/null 2>&1; then
    return
  fi

  PASSWORD_FILE="$(mktemp)"
  chmod 600 "$PASSWORD_FILE"
  printf '%s\n' "$password" > "$PASSWORD_FILE"
  unset password
}

ssh_cmd() {
  if [ -n "$PASSWORD_FILE" ]; then
    sshpass -f "$PASSWORD_FILE" ssh \
      -o PreferredAuthentications=publickey,password \
      -o PubkeyAuthentication=yes \
      "$@"
  else
    ssh "$@"
  fi
}

rsync_cmd() {
  if [ -n "$PASSWORD_FILE" ]; then
    sshpass -f "$PASSWORD_FILE" rsync \
      -e "ssh -o PreferredAuthentications=publickey,password -o PubkeyAuthentication=yes" \
      "$@"
  else
    rsync "$@"
  fi
}

sync_code() {
  local node="$1"
  local local_code remote_parent
  local_code="$(nflownet_root)"
  remote_parent="$(dirname "$REMOTE_CODE_DIR")"

  echo "Syncing code: ${local_code}/ -> ${node}:${REMOTE_CODE_DIR}/"
  if [ "$DRY_RUN" = "1" ]; then
    if [ "$TRAIN_VARIANT" = "raft" ]; then
      echo "Would export RAFT trainer: ${REMOTE_CODE_DIR}/train_raft.py -> ${REMOTE_CODE_DIR}/train.py"
    fi
    return
  fi

  ssh_cmd "$node" "mkdir -p '$remote_parent' '$REMOTE_CODE_DIR'"
  rsync_cmd -az --delete \
    --exclude ".git/" \
    --exclude ".venv/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "logs/" \
    --exclude "runs/" \
    --exclude "checkpoints/" \
    "${local_code}/" "${node}:${REMOTE_CODE_DIR}/"

  if [ "$TRAIN_VARIANT" = "raft" ]; then
    echo "Exporting RAFT trainer: ${REMOTE_CODE_DIR}/train_raft.py -> ${REMOTE_CODE_DIR}/train.py"
    ssh_cmd "$node" "test -f '${REMOTE_CODE_DIR}/train_raft.py' && cp '${REMOTE_CODE_DIR}/train_raft.py' '${REMOTE_CODE_DIR}/train.py'"
  fi
}

sync_data() {
  local node="$1"
  local local_data
  local_data="$(abs_or_workspace_path "$LOCAL_DATA_DIR")"

  if [ ! -d "$local_data" ]; then
    echo "Local TartanAir directory does not exist: $local_data" >&2
    exit 1
  fi

  echo "Syncing TartanAir: ${local_data}/ -> ${node}:${REMOTE_DATA_DIR}/"
  if [ "$DRY_RUN" = "1" ]; then
    return
  fi

  ssh_cmd "$node" "mkdir -p '$REMOTE_DATA_DIR'"
  rsync_cmd -az --delete --partial --info=progress2 \
    "${local_data}/" "${node}:${REMOTE_DATA_DIR}/"
}

run_remote() {
  local node="$1"

  echo "${ACTION}: ${node}"
  if [ "$DRY_RUN" = "1" ]; then
    return
  fi

  ssh_cmd "$node" "bash -s" -- \
    "$ACTION" "$REMOTE_CODE_DIR" "$REMOTE_DATA_DIR" "$REMOTE_VENV" "$REMOTE_LOG_DIR" "$TRAIN_CMD_B64" "$REMOTE_ROOT" "$FOLLOW_LOGS" "$TRAIN_VARIANT" <<'REMOTE'
set -euo pipefail

action="$1"
code_arg="$2"
data_arg="$3"
venv_arg="$4"
log_arg="$5"
train_cmd_b64="$6"
root_arg="$7"
follow_logs="$8"
train_variant="$9"

resolve_home_path() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$HOME" "$1"
  fi
}

code_dir="$(resolve_home_path "$code_arg")"
data_dir="$(resolve_home_path "$data_arg")"
root_dir="$(resolve_home_path "$root_arg")"

path_from_code_dir() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$code_dir" "$1"
  fi
}

venv_path="$(path_from_code_dir "$venv_arg")"
log_dir="$(path_from_code_dir "$log_arg")"
pid_file="${log_dir}/train.pid"
log_file="${log_dir}/train.log"
train_cmd="$(printf '%s' "$train_cmd_b64" | base64 --decode)"

prepare_venv() {
  if [ ! -x "${venv_path}/bin/python" ]; then
    echo "Creating remote virtualenv: ${venv_path}"
    python3 -m venv --system-site-packages "$venv_path"
  fi

  if ! "${venv_path}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
import torchvision
PY
  then
    echo "Installing remote Python packages: torch torchvision numpy"
    "${venv_path}/bin/python" -m pip install --upgrade pip
    "${venv_path}/bin/python" -m pip install torch torchvision numpy
  fi
}

select_training_variant() {
  case "$train_variant" in
    train)
      ;;
    raft)
      if [ ! -f "${code_dir}/train_raft.py" ]; then
        echo "Missing RAFT trainer: ${code_dir}/train_raft.py" >&2
        exit 1
      fi
      cp "${code_dir}/train_raft.py" "${code_dir}/train.py"
      echo "Selected RAFT trainer: train_raft.py -> train.py"
      ;;
    *)
      echo "Unknown training variant: $train_variant" >&2
      exit 2
      ;;
  esac
}

set_training_root() {
  local train_file="${code_dir}/train.py"
  python3 - "$root_dir" "$train_file" <<'PY'
from pathlib import Path
import re
import sys

root = sys.argv[1]
train_file = Path(sys.argv[2])
source = train_file.read_text()
updated, count = re.subn(
    r'(?m)^ROOT\s*=.*$',
    f'ROOT = {root!r}',
    source,
    count=1,
)
if count != 1:
    raise SystemExit(f"Could not find a top-level ROOT assignment in {train_file}")
train_file.write_text(updated)
print(f"Set remote train.py ROOT = {root}")
PY
}

case "$action" in
  start)
    test -d "$code_dir"
    test -f "${code_dir}/train.py"
    if [ ! -d "$data_dir" ]; then
      echo "Remote TartanAir directory is missing: $data_dir" >&2
      echo "Run: NFlowNet/scripts/dgx_train.sh sync-data --node 1" >&2
      exit 1
    fi
    cd "$code_dir"
    select_training_variant
    set_training_root
    prepare_venv
    mkdir -p "$log_dir"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "already running with pid $(cat "$pid_file")"
      echo "log: $log_file"
      exit 0
    fi
    export PYTHONUNBUFFERED=1
    export ROOT="$root_dir"
    export TARTANAIR_ROOT="$data_dir"
    nohup bash -lc "source '${venv_path}/bin/activate' && ${train_cmd}" > "$log_file" 2>&1 < /dev/null &
    echo "$!" > "$pid_file"
    echo "started pid $(cat "$pid_file")"
    echo "log: $log_file"
    ;;
  status)
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "running with pid $(cat "$pid_file")"
    else
      echo "not running"
    fi
    if [ -f "$log_file" ]; then
      echo "last log lines from $log_file:"
      tail -30 "$log_file"
    fi
    ;;
  tail)
    if [ -f "$log_file" ]; then
      if [ "$follow_logs" = "1" ]; then
        tail -n 100 -f "$log_file"
      else
        tail -100 "$log_file"
      fi
    else
      echo "no log file yet: $log_file"
    fi
    ;;
  stop)
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      kill "$(cat "$pid_file")"
      echo "stopped pid $(cat "$pid_file")"
    else
      echo "not running"
    fi
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac
REMOTE
}

mapfile -t nodes < <(load_nodes)
init_password_auth

echo "DGX target(s):"
printf '  %s\n' "${nodes[@]}"
echo "Remote code: ${REMOTE_CODE_DIR}"
echo "Remote data: ${REMOTE_DATA_DIR}"
echo "Training variant: ${TRAIN_VARIANT}"
echo

case "$ACTION" in
  start)
    for node in "${nodes[@]}"; do
      if [ "$SYNC_CODE" = "1" ]; then
        sync_code "$node"
      fi
      if [ "$SYNC_DATA" = "1" ]; then
        sync_data "$node"
      fi
      run_remote "$node"
    done
    ;;
  sync-code)
    for node in "${nodes[@]}"; do
      sync_code "$node"
    done
    ;;
  sync-data)
    for node in "${nodes[@]}"; do
      sync_data "$node"
    done
    ;;
  status|tail|stop)
    for node in "${nodes[@]}"; do
      run_remote "$node"
    done
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
