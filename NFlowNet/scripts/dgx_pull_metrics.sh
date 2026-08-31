#!/usr/bin/env bash
set -euo pipefail

NODE_SELECTOR="${DGX_NODE:-1}"
REMOTE_CODE_DIR="${DGX_REMOTE_CODE_DIR:-Optical_Flow/NFlowNet}"
LOCAL_OUTPUT_DIR="${DGX_LOCAL_OUTPUT_DIR:-NFlowNet}"
ACCESS_FILE="${DGX_ACCESS_FILE:-$HOME/dgx_access}"
AUTO_PASSWORD="${DGX_AUTO_PASSWORD:-1}"
DRY_RUN=0
PASSWORD_FILE=""

usage() {
  cat <<USAGE
Usage:
  NFlowNet/scripts/dgx_pull_metrics.sh [options]

Options:
  --node 1|2          DGX node source. Default: ${NODE_SELECTOR}
  --remote-code PATH  Remote NFlowNet path, relative to \$HOME unless absolute.
                      Default: ${REMOTE_CODE_DIR}
  --output-dir PATH   Local destination, relative to workspace root unless absolute.
                      Default: ${LOCAL_OUTPUT_DIR}
  --dry-run           Print the rsync command without running it.
  -h, --help          Show this help.
USAGE
}

cleanup_password_file() {
  if [ -n "$PASSWORD_FILE" ] && [ -f "$PASSWORD_FILE" ]; then
    rm -f "$PASSWORD_FILE"
  fi
}
trap cleanup_password_file EXIT

while [ "$#" -gt 0 ]; do
  case "$1" in
    --node)
      NODE_SELECTOR="$2"
      shift 2
      ;;
    --remote-code)
      REMOTE_CODE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      LOCAL_OUTPUT_DIR="$2"
      shift 2
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

workspace_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
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
    *)
      echo "--node must be 1 or 2" >&2
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

rsync_cmd() {
  if [ -n "$PASSWORD_FILE" ]; then
    sshpass -f "$PASSWORD_FILE" rsync \
      -e "ssh -o PreferredAuthentications=publickey,password -o PubkeyAuthentication=yes" \
      "$@"
  else
    rsync "$@"
  fi
}

node="$(load_nodes)"
local_output="$(abs_or_workspace_path "$LOCAL_OUTPUT_DIR")"

echo "Pulling metrics from ${node}:${REMOTE_CODE_DIR}/"
echo "Local output: ${local_output}/"

if [ "$DRY_RUN" = "1" ]; then
  echo "rsync -az ${node}:${REMOTE_CODE_DIR}/{training_metrics.json,validation_metrics.json} ${local_output}/"
  exit 0
fi

init_password_auth
mkdir -p "$local_output"
rsync_cmd -az \
  "${node}:${REMOTE_CODE_DIR}/training_metrics.json" \
  "${node}:${REMOTE_CODE_DIR}/validation_metrics.json" \
  "${local_output}/"
