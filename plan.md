# autoresearch — agent playbook (read this first)

You are GitHub Copilot, running inside an **airgapped** container managed by
[`msr-ai4science/airgap`](https://github.com/msr-ai4science/airgap). The
container has **no general internet access** — only the airgap proxies are
reachable. Everything you need to do research is already on disk.

## What is pre-staged for you

| Artifact | Location | Status |
|---|---|---|
| Python venv (torch+cu128, kernels, tiktoken, rustbpe, …) | `/opt/autoresearch/.venv` (managed by `uv`) | Frozen at image build |
| Flash-Attention-3 CUDA kernels | `/opt/hf-cache/` | Frozen, accessed via `LOCAL_KERNELS` |
| Training data shards + BPE tokenizer | `/data/` (env: `AUTORESEARCH_CACHE_DIR=/data`) | Populated by the operator before launch |
| Project source (you may edit `train.py`) | `/opt/autoresearch/` | Mounted from host (`WORK_DIR`) |
| Loop semantics | `program.md` (next to this file) | Read after the verification step |

## Step 1 — Verify the environment

Run, in this order, and stop with a clear error if any step fails:

```bash
nvidia-smi | head -20                           # GPU visible?
echo "$LOCAL_KERNELS"                           # non-empty?
echo "$AUTORESEARCH_CACHE_DIR"                  # /data
ls "$AUTORESEARCH_CACHE_DIR/data" | wc -l       # >= 2 parquet shards
ls "$AUTORESEARCH_CACHE_DIR/tokenizer"          # tokenizer.pkl + token_bytes.pt
cd /opt/autoresearch && uv run python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

If `AUTORESEARCH_CACHE_DIR/data` is empty, **stop and report** — the operator
forgot to run the host-side prepare step (see README). Do **not** attempt to
download data yourself; the network will not let you.

## Step 2 — Smoke-test the training loop

```bash
cd /opt/autoresearch
uv run train.py > /tmp/smoke.log 2>&1
grep -E "^val_bpb:|^peak_vram_mb:" /tmp/smoke.log
```

Expect a `val_bpb:` line within ~6 minutes. If you don't see one, `tail -n 50
/tmp/smoke.log` and report the failure.

## Step 3 — Run the experiment loop forever

Read [`program.md`](./program.md) and follow it literally — it defines the
branch / edit / run / grep / keep-or-revert loop, the `results.tsv` schema,
and the "NEVER STOP" rule. The only adaptation for the airgap container:

- All `uv run train.py` invocations should redirect to `run.log` exactly as
  `program.md` specifies (do not `tee`, do not let output flood your context).
- You **cannot** install new packages, push to a remote, or fetch from the
  internet. Stay on the local branch and only commit locally.
- If `train.py` crashes with a kernel-not-found or HuggingFace error, the
  `LOCAL_KERNELS` env var is misconfigured — stop and report; do not retry.

Now: read `program.md` and begin.
