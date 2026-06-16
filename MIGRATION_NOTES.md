# Migration notes — Qwen-Image-Edit-2511 LoRA fine-tuning, WSL2 → native Windows 11

## Why we migrated

The previous training attempt under WSL2 died at `accelerator.prepare()` with
`ncclUnhandledCudaError` / `Cuda failure 999 'unknown error'` — captured in
`error.txt`. Root cause is the WSL2 WDDM driver path losing CUDA context when
DDP forks worker processes. NCCL multi-GPU on WSL2 is fragile by design; the
fix is to leave WSL2 entirely.

## Why DiffSynth-Studio is the primary trainer

- The Qwen team officially endorses DiffSynth-Studio for fine-tuning their
  image models.
- It ships a tuned LoRA recipe for Qwen-Image-Edit-2511 at
  `examples/qwen_image/model_training/train.py`.
- It exposes `--initialize_model_on_cpu`, which is the documented fix for the
  load-on-GPU-before-prepare DDP hang (see "Critical fixes" below).
- The legacy diffusers-based script under `src/train_qwen_edit_lora/` is kept
  as a fallback and was ported in parallel.

DiffSynth-Studio is **pinned at commit
`afd101f3452c9ecae0c87b79adfa2e22d65ffdc3`**. The repo evolves daily; without
pinning, the next clone may not behave identically. To reproduce, after
cloning the upstream `https://github.com/modelscope/DiffSynth-Studio.git`, run
`git checkout afd101f3452c9ecae0c87b79adfa2e22d65ffdc3`.

## Why the 4-bit base — and why we did NOT drop bitsandbytes

The original migration plan said "drop bitsandbytes, run full-precision Qwen-
Image-Edit-2511". The VRAM math says otherwise:

| Component                                 | bf16    | 4-bit |
| ----------------------------------------- | ------: | ----: |
| Frozen transformer (~20.5 B params)       | 41 GB   | 11 GB |
| Activations @ 768 res w/ grad checkpoint  | ~7 GB   | ~7 GB |
| Text encoder + VAE on GPU                 | ~4 GB   | ~4 GB |
| LoRA adapters + optimizer state (47 M)    | ~0.2 GB | ~0.2 GB |
| **Total**                                 | **~52 GB** | **~22 GB** |

Full-precision does not fit on a 24 GB A5000. We keep the
`toandev/Qwen-Image-Edit-2511-4bit` fork pinned at revision
`bbf3075e1aee61f91027fa212c9a3a03cd5dc7c4`. bitsandbytes is required only to
*load* the 4-bit base; the optimizer is plain `torch.optim.AdamW` (LoRA's
optimizer state is ~0.1 GB, so 8-bit AdamW saves nothing).

Resolution stays at **768**, not 1024. 1024 lives at the OOM ceiling.

## Why gloo, not NCCL

PyTorch's Windows wheels ship `gloo` only — NCCL has no Windows build. Gloo
is slower than NCCL on large allreduce, but for LoRA-only training the
synced tensors are ~47 M params (~100 MB / step), which gloo handles fine.

Two non-obvious flags are required for `accelerate launch` on Windows:

- `--rdzv_backend c10d`. The default `static` backend's TCPStore requests
  libuv unconditionally, and the Windows wheel was built without libuv. With
  `static`, multi-GPU launch fails immediately with
  `RuntimeError: use_libuv was requested but PyTorch was build without libuv support`.
  `c10d` rendezvous uses a different code path that respects `USE_LIBUV=0`.
- `USE_LIBUV=0` env var (set in `set_env.ps1`). Even with `c10d`, some torch
  internals will try libuv unless this is set.

Also: do **not** set `GLOO_SOCKET_IFNAME=Loopback` on Windows. Gloo's UV
transport rejects "Loopback" as an iface name (`Unable to find address for:
Loopback`) — letting it auto-pick uses the default v4 iface.

## Critical fix: load model on CPU before `accelerator.prepare()`

The WSL2 NCCL failure has a deeper root cause that *survives* migration to
gloo if not fixed: in `src/train_qwen_edit_lora/train_qwen_edit_lora.py`,
the trainable transformer was being moved to GPU via `transformer.to(...)`
*before* `accelerator.prepare()` wrapped it in DDP. Under gloo, DDP's
`_verify_param_shape_across_processes` allgather requires the params to be
reachable through the chosen backend; if they're already on CUDA and the
wrong device, this hangs.

Fix in the legacy script: removed the explicit `.to(accelerator.device)`
call. `accelerator.prepare()` moves the model to the right device.

Equivalent for DiffSynth: pass `--initialize_model_on_cpu` on the CLI.

The legacy script also now passes
`kwargs_handlers=[InitProcessGroupKwargs(backend="gloo")]` on Windows — this
overrides accelerate's default of NCCL when CUDA is visible.

## Pinned dependency versions (Windows)

| Package          | Pinned              | Why                                      |
| ---------------- | ------------------- | ---------------------------------------- |
| Python           | 3.11                | Best wheel coverage on Windows           |
| torch            | 2.4.1+cu124         | Stable diffusers compat; no torch 2.5+   |
| transformers     | 4.57.6 (>=4.49,<5)  | 4.49+ has `Qwen2_5_VLModel`; <5 keeps `PretrainedConfig` in `modeling_utils` (DiffSynth depends on this) |
| diffusers        | 0.36.0              | First version with `QwenImageEditPlusPipeline`; 0.37+ requires torch>=2.5 |
| accelerate       | 0.34.2              | Compatible with our peft/diffusers pins  |
| peft             | 0.19.1              | diffusers 0.36 requires peft>=0.17        |
| bitsandbytes     | 0.43.3              | Loads the 4-bit base                     |
| huggingface_hub  | 0.36.x              | Compatible with transformers 4.57         |

Plan-vs-actual deviations:
- transformers: planned 4.45.2 → actual 4.57.6. 4.45 is too old for
  `Qwen2_5_VLModel`. The DiffSynth pinned commit needs >=4.49 but <5.
- diffusers: planned 0.31.0 → actual 0.36.0. 0.31 lacks
  `QwenImageEditPlusPipeline`; 0.37+ has a flash_attn_3 dispatch that
  fails to register under torch 2.4.1.
- peft: planned 0.13.0 → actual 0.19.1. 0.13 doesn't satisfy diffusers 0.36's
  `peft>=0.17` requirement.
- setuptools: had to downgrade to <81 (we use 80.10.2). Setuptools 81+
  removes `pkg_resources`, which DiffSynth's `setup.py` still imports.

## Multi-GPU vs parallel-HP-sweep alternative

We chose multi-GPU (4× A5000 via gloo) per the user's preference. There is a
sharper alternative worth keeping in mind: run **4 independent single-GPU
jobs in parallel**, one per GPU, each with different hyperparameters
(LR / rank / sigma sampling). For LoRA training on a small dataset, gloo's
allreduce overhead eats most of the per-step speedup, so the wall time saved
by going multi-GPU vs. running a single GPU is small. Parallel HP sweeps
give 4× experimental throughput and zero distributed-training risk.

If multi-GPU starts misbehaving for non-obvious reasons, fall back to:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
.\.venv-win\Scripts\python.exe ..\DiffSynth-Studio\examples\qwen_image\model_training\train.py <args>
# (no `accelerate launch`)
```

DiffSynth's `Accelerator()` will see one device and run pure single-GPU.

## Recommended (NOT auto-applied) Defender exclusions

Run from an **elevated** PowerShell:

```powershell
Add-MpPreference -ExclusionPath "E:\FinetuningQwen"
Add-MpPreference -ExclusionPath "E:\FinetuningQwen\hf_cache"
```

Real-time scanning of the safetensors shards across 4 simultaneous loads
triples wall time and occasionally races c10d rendezvous. Not auto-applied
because it modifies system security policy.

## Out of scope (deferred)

- **Dataset preparation**: PDF → (control, target, caption) triplet pipeline.
  The CSV is being provided separately. When it arrives, the next planning
  session will design the page-extraction strategy, the student↔teacher
  page matching, the caption generation method, and a 200-sample pilot.
- **`qwen_server_lora.py` LoRA loading wiring**: still not connected; the
  active server only loads the base pipeline.
- **Inference at full-precision**: `inference_lora.py` defaults to
  `Qwen/Qwen-Image-Edit-2511` (full precision). Loading 41 GB on a 24 GB
  A5000 requires CPU offload — not yet wired.
