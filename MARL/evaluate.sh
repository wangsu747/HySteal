set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
ENV_TYPE="${ENV_TYPE:-simple_tag_v3}"

case "$ENV_TYPE" in
  simple_tag_v3)
    "$PY" "$ROOT_DIR/evaluate.py" "$@"
    ;;
  grf)
    "$PY" "$ROOT_DIR/evaluate.py" "$@"
    ;;
  overcooked)
    "$PY" "$ROOT_DIR/evaluate.py" "$@"
    ;;
  *)
    echo "Unsupported ENV_TYPE: $ENV_TYPE"
    exit 1
    ;;
 esac
