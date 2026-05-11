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

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

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
RUN pip3 install --no-cache-dir --break-system-packages uv==0.5.*

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
    fi

# Resolve the pinned snapshot SHAs and bake them into LOCAL_KERNELS, which
# short-circuits get_kernel() to a pure filesystem path at runtime — this is
# the only fully reliable way to be offline (HF_HUB_OFFLINE=1 alone does NOT
# work because kernels.list_repo_tree() is a live REST call).
RUN COMMUNITY_SHA=$(cat /opt/hf-cache/kernels--kernels-community--flash-attn3/refs/main 2>/dev/null) ; \
    VARUNNEAL_SHA=$(cat /opt/hf-cache/kernels--varunneal--flash-attention-3/refs/main 2>/dev/null) ; \
    { \
      echo "export HF_HUB_CACHE=/opt/hf-cache" ; \
      echo "export KERNELS_CACHE=/opt/hf-cache" ; \
      echo "export HF_HUB_OFFLINE=1" ; \
      echo "export DISABLE_TELEMETRY=yes" ; \
      echo "export AUTORESEARCH_CACHE_DIR=/data" ; \
      if [ -n "$COMMUNITY_SHA" ] && [ -n "$VARUNNEAL_SHA" ]; then \
        echo "export LOCAL_KERNELS=\"kernels-community/flash-attn3=/opt/hf-cache/kernels--kernels-community--flash-attn3/snapshots/$COMMUNITY_SHA:varunneal/flash-attention-3=/opt/hf-cache/kernels--varunneal--flash-attention-3/snapshots/$VARUNNEAL_SHA\"" ; \
      elif [ -n "$COMMUNITY_SHA" ]; then \
        echo "export LOCAL_KERNELS=\"kernels-community/flash-attn3=/opt/hf-cache/kernels--kernels-community--flash-attn3/snapshots/$COMMUNITY_SHA\"" ; \
      elif [ -n "$VARUNNEAL_SHA" ]; then \
        echo "export LOCAL_KERNELS=\"varunneal/flash-attention-3=/opt/hf-cache/kernels--varunneal--flash-attention-3/snapshots/$VARUNNEAL_SHA\"" ; \
      fi ; \
    } > /etc/profile.d/autoresearch.sh && chmod 0644 /etc/profile.d/autoresearch.sh

# Mirror the same env into the system-wide environment so non-login shells
# (which is how Copilot's `bash -c` invocations run) also see them.
RUN sed -E 's/^export +//' /etc/profile.d/autoresearch.sh >> /etc/environment

# Gap 4: data lives on a host-mounted volume at /data (airgap --mount).
# The agent reads/writes here via AUTORESEARCH_CACHE_DIR=/data (see env above).
# The volume is populated once on the host via:
#   docker run --rm --gpus all \
#     -v /host/path/data:/data -e AUTORESEARCH_CACHE_DIR=/data \
#     <this-image> bash -lc "cd /opt/autoresearch && uv run prepare.py --num-shards 10"
RUN mkdir -p /data
VOLUME ["/data"]

# airgap creates user `airgap` with UID/GID 1000 in Stage 2. Pre-chown the
# project + kernel cache so that user can read everything without root.
RUN chown -R 1000:1000 /opt/autoresearch /opt/hf-cache /data
