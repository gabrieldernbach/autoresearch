# Stage-1 base image for autoresearch, designed to be wrapped by msr-ai4science/airgap.
#
# What this image contains (the "trusted user space"):
#   * CUDA 12.8 + cuDNN runtime on Ubuntu 24.04
#   * Python 3.10 + uv 0.5.x
#   * The full pinned .venv from uv.lock (torch==2.9.1+cu128 and friends)
#   * Pre-staged Flash-Attention-3 CUDA kernel binaries in /opt/hf-cache, with
#     LOCAL_KERNELS configured so train.py never reaches HuggingFace at runtime
#
# What this image does NOT contain:
#   * Training data shards / tokenizer (host-mounted at /data via airgap --mount,
#     populated once with `docker run ... uv run prepare.py` — see README)
#   * Node.js / @github/copilot (added on top by airgap's Stage 2 wrapper)
#   * ENTRYPOINT (airgap clears any ENTRYPOINT in Stage 2)
#
# The image must remain Debian/Ubuntu-based (airgap rejects Alpine).

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        python3.10 python3.10-venv python3.10-dev python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Gap 1: vendor uv inside the image (no `curl astral.sh | sh` at runtime)
RUN pip3 install --no-cache-dir uv==0.5.*

WORKDIR /opt/autoresearch

# Copy dependency manifests first so the heavy `uv sync` layer caches well.
COPY pyproject.toml uv.lock .python-version ./
COPY kernels.lock* ./

# Gap 2: install all Python deps (PyPI + the explicit pytorch-cu128 index)
# into a frozen .venv inside the image. Requires internet at build time.
RUN uv sync --frozen --no-install-project

# Now copy the actual project source.
COPY *.py program.md plan.md README.md ./

# Reinstall to register the project itself (cheap, deps already cached).
RUN uv sync --frozen

# Gap 3: pre-stage Flash-Attention-3 CUDA kernels into a fixed cache dir.
# We use a project-local cache (/opt/hf-cache) so we can chown it cleanly.
# `--all-variants` ensures the image works on both Hopper (H100) and Ampere
# (A100) without an additional download. If kernels.lock is absent, we fall
# back to a direct get_kernel() warm-up for both repos.
ENV HF_HUB_CACHE=/opt/hf-cache \
    KERNELS_CACHE=/opt/hf-cache
RUN mkdir -p /opt/hf-cache && \
    if [ -f kernels.lock ]; then \
        uv run kernels download . --all-variants ; \
    else \
        uv run python -c "from kernels import get_kernel; \
get_kernel('kernels-community/flash-attn3'); \
get_kernel('varunneal/flash-attention-3')" ; \
    fi && \
    # Create stable 'current' symlinks so ENV LOCAL_KERNELS below can be static.
    for repo in kernels--kernels-community--flash-attn3 kernels--varunneal--flash-attention-3 ; do \
      sha=$(cat /opt/hf-cache/$repo/refs/main 2>/dev/null) ; \
      if [ -n "$sha" ] && [ -d /opt/hf-cache/$repo/snapshots/$sha ]; then \
        ln -sfn "$sha" /opt/hf-cache/$repo/snapshots/current ; \
      fi ; \
    done

# Static env vars baked at the Docker layer so they apply to non-login shells
# (Copilot launches via `bash -c`, which sources neither /etc/profile.d nor
# /etc/environment). LOCAL_KERNELS short-circuits get_kernel() to a pure
# filesystem path — the only fully reliable way to be offline (HF_HUB_OFFLINE=1
# alone does NOT suffice because kernels.list_repo_tree() is a live REST call).
ENV HF_HUB_OFFLINE=1 \
    DISABLE_TELEMETRY=yes \
    AUTORESEARCH_CACHE_DIR=/data/autoresearch \
    LOCAL_KERNELS="kernels-community/flash-attn3=/opt/hf-cache/kernels--kernels-community--flash-attn3/snapshots/current:varunneal/flash-attention-3=/opt/hf-cache/kernels--varunneal--flash-attention-3/snapshots/current"

# Gap 4: data lives on a host-mounted volume at /data/autoresearch (identity-
# mapped via `airgap --mount /data/autoresearch`). The agent reads/writes here
# via AUTORESEARCH_CACHE_DIR=/data/autoresearch (set in ENV above). The
# volume is populated once on the host with:
#   docker run --rm --gpus all \
#     -v /data/autoresearch:/data/autoresearch \
#     -e AUTORESEARCH_CACHE_DIR=/data/autoresearch \
#     <this-image> bash -lc "cd /opt/autoresearch && uv run prepare.py --num-shards 10"
RUN mkdir -p /data/autoresearch
VOLUME ["/data/autoresearch"]

# NOTE: airgap's Stage 2 wrapper creates the agent user with the HOST UID/GID,
# not necessarily 1000. We rely on world-readable defaults (0755 dirs, 0644
# files, 0755 binaries) so that any UID can read /opt/autoresearch and
# /opt/hf-cache. The /data volume is bind-mounted from the host with host
# ownership, so it is naturally writable by the agent user.
