#!/usr/bin/env bash
# start.sh — bring up autoresearch inside an airgapped Copilot container.
#
# Assumes:
#   * `airgap` is already installed and `airgap auth login` has been run
#     (https://github.com/msr-ai4science/airgap)
#   * Docker 23+ with the NVIDIA Container Toolkit
#   * An NVIDIA GPU (H100 tested; A100 supported)
#
# This script is fully idempotent — re-running it after the first launch
# only re-invokes the agent (steps 1–4 are no-ops if already done).
#
# Usage:
#   ./start.sh                   # default prompt: "@plan.md run the plan"
#   ./start.sh "do X then Y"     # custom prompt forwarded to copilot -p

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${AUTORESEARCH_PROJECT:-autoresearch}"
DATA_DIR="${AUTORESEARCH_DATA_DIR:-/data/autoresearch}"
NUM_SHARDS="${AUTORESEARCH_NUM_SHARDS:-10}"
PROMPT="${1:-@plan.md run the plan}"

log() { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. Sanity
# ---------------------------------------------------------------------------
command -v airgap >/dev/null || { echo "airgap not installed; see README"; exit 1; }
command -v docker >/dev/null || { echo "docker not installed"; exit 1; }
docker info >/dev/null 2>&1   || { echo "docker daemon unreachable"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Host data directory (mounted into the agent container at /data)
# ---------------------------------------------------------------------------
if [[ ! -d "$DATA_DIR" ]]; then
  log "creating $DATA_DIR"
  if ! mkdir -p "$DATA_DIR" 2>/dev/null; then
    sudo mkdir -p "$DATA_DIR"
    sudo chown "$USER" "$DATA_DIR"
  fi
fi

# ---------------------------------------------------------------------------
# 2. Register the project with airgap (idempotent — skips if config exists)
# ---------------------------------------------------------------------------
CONF="$HOME/.config/airgap/${PROJECT_NAME}.conf"
if [[ ! -f "$CONF" ]]; then
  log "airgap init $PROJECT_NAME"
  airgap init "$PROJECT_NAME" \
    --dockerfile "$REPO_DIR/Dockerfile" \
    --work-dir   "$REPO_DIR" \
    --mount      "$DATA_DIR" \
    --allow      "huggingface.co cdn-lfs.huggingface.co cdn-lfs-us-1.huggingface.co download.pytorch.org"
else
  log "airgap config exists at $CONF (skipping init)"
fi

# ---------------------------------------------------------------------------
# 3. Build the project + airgap wrapper images (fingerprint-cached, fast on re-run)
# ---------------------------------------------------------------------------
log "airgap $PROJECT_NAME build"
airgap "$PROJECT_NAME" build

# ---------------------------------------------------------------------------
# 4. One-time host-side prepare (trusted user space, with internet).
#    prepare.py is idempotent: per-shard exists-check + tokenizer skip-if-exists.
# ---------------------------------------------------------------------------
NEED_PREPARE=0
if [[ ! -f "$DATA_DIR/tokenizer/tokenizer.pkl" ]] || \
   [[ ! -f "$DATA_DIR/tokenizer/token_bytes.pt" ]] || \
   [[ -z "$(ls -A "$DATA_DIR/data" 2>/dev/null || true)" ]]; then
  NEED_PREPARE=1
fi

if [[ "$NEED_PREPARE" == "1" ]]; then
  log "host-side prepare: docker run prepare.py --num-shards $NUM_SHARDS"
  docker run --rm --gpus all \
    -v "$DATA_DIR":/data \
    -e AUTORESEARCH_CACHE_DIR=/data \
    "airgap-${PROJECT_NAME}-base" \
    bash -lc "cd /opt/autoresearch && uv run prepare.py --num-shards $NUM_SHARDS"
else
  log "data and tokenizer already present at $DATA_DIR (skipping prepare)"
fi

# ---------------------------------------------------------------------------
# 5. Hand off to Copilot, fully airgapped
# ---------------------------------------------------------------------------
log "airgap $PROJECT_NAME copilot -p \"$PROMPT\""
exec airgap "$PROJECT_NAME" copilot -p "$PROMPT"
