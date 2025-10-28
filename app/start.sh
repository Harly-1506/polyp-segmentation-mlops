#!/bin/bash
set -euo pipefail

# Always execute relative to this script's directory so that local and in-container
# executions resolve the same module paths (e.g. `backend.main`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Utility to normalise various truthy/falsey env values to "true"/"false".
to_bool() {
  case "${1:-}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Yy])
      echo "true"
      ;;
    *)
      echo "false"
      ;;
  esac
}

BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
BACKEND_PORT=${BACKEND_PORT:-8081}
BACKEND_LOG_LEVEL=${BACKEND_LOG_LEVEL:-info}

RUN_BACKEND=$(to_bool "${RUN_BACKEND:-true}")
# Preserve backwards compatibility with ENABLE_GRADIO_UI while allowing explicit overrides.
DEFAULT_RUN_UI=$(to_bool "${ENABLE_GRADIO_UI:-false}") # Default if RUN_UI is not set. change to true if run locally
RUN_UI=$(to_bool "${RUN_UI:-$DEFAULT_RUN_UI}")

UI_HOST=${UI_HOST:-0.0.0.0}
UI_PORT=${UI_PORT:-${GRADIO_SERVER_PORT:-7860}}

if [[ -z "${PORT:-}" ]]; then
  if [[ "$RUN_UI" == "true" ]]; then
    export PORT="$UI_PORT"
  elif [[ "$RUN_BACKEND" == "true" ]]; then
    export PORT="$BACKEND_PORT"
  else
    export PORT="$UI_PORT"
  fi
fi
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-$UI_HOST}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-$UI_PORT}"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"

run_inference_group() {
  if command -v uv >/dev/null 2>&1; then
    uv run --group inference -- "$@"
  else
    "$@"
  fi
}

launch_backend() {
  local module_path="$1"
  run_inference_group uvicorn "$module_path" \
    --host "$BACKEND_HOST" \
    --port "$BACKEND_PORT" \
    --log-level "$BACKEND_LOG_LEVEL" &
  BACKEND_PID=$!
  echo "Backend started on ${BACKEND_HOST}:${BACKEND_PORT} (PID ${BACKEND_PID})" >&2
}

launch_gradio() {
  local entry_point="$1"
  run_inference_group python "$entry_point" &
  GRADIO_PID=$!
  echo "Gradio UI started on ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT} (PID ${GRADIO_PID})" >&2
}

shutdown() {
  trap - EXIT
  if [[ -n "${GRADIO_PID:-}" ]]; then
    kill "$GRADIO_PID" 2>/dev/null || true
    wait "$GRADIO_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap shutdown EXIT TERM INT

BACKEND_PID=""
GRADIO_PID=""

if [[ "$RUN_BACKEND" == "true" ]]; then
  BACKEND_APP_MODULE=${BACKEND_APP_MODULE:-}
  if [[ -z "$BACKEND_APP_MODULE" ]]; then
    if [[ -f "backend/main.py" ]]; then
      BACKEND_APP_MODULE="backend.main:app"
    elif [[ -f "app/backend/main.py" ]]; then
      BACKEND_APP_MODULE="app.backend.main:app"
    else
      echo "Unable to determine backend module. Set BACKEND_APP_MODULE explicitly." >&2
      exit 1
    fi
  fi
  launch_backend "$BACKEND_APP_MODULE"
fi

if [[ "$RUN_UI" == "true" ]]; then
  UI_ENTRYPOINT=${UI_ENTRYPOINT:-}
  if [[ -z "$UI_ENTRYPOINT" ]]; then
    if [[ -f "frontend/gradio_ui.py" ]]; then
      UI_ENTRYPOINT="frontend/gradio_ui.py"
    elif [[ -f "app/frontend/gradio_ui.py" ]]; then
      UI_ENTRYPOINT="app/frontend/gradio_ui.py"
    else
      echo "Unable to determine Gradio UI entrypoint. Set UI_ENTRYPOINT explicitly." >&2
      exit 1
    fi
  fi
  launch_gradio "$UI_ENTRYPOINT"
fi

if [[ -n "${BACKEND_PID:-}" && -n "${GRADIO_PID:-}" ]]; then
  wait "$BACKEND_PID" "$GRADIO_PID"
elif [[ -n "${BACKEND_PID:-}" ]]; then
  wait "$BACKEND_PID"
elif [[ -n "${GRADIO_PID:-}" ]]; then
  wait "$GRADIO_PID"
else
  echo "Nothing to run. Set RUN_BACKEND=true and/or RUN_UI=true." >&2
  exit 1
fi