set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

if [[ $ -lt 2 ]]; then
  echo "Usage: ./evaluate.sh <env> <ckpt_path> [extra args...]"
  exit 1
fi

ENV_NAME="$1"
CKPT_PATH="$2"
shift 2

"$PY" "$ROOT_DIR/magail/evaluate.py" --env "$ENV_NAME" --ckpt_path "$CKPT_PATH" "$@"
