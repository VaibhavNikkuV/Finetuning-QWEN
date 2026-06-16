# Legacy diffusers-based training, multi-GPU on Windows (gloo).
# Use this if DiffSynth-Studio's recipe has issues; otherwise prefer train_diffsynth.ps1.

$ErrorActionPreference = "Stop"

# Source env (HF_HOME, USE_LIBUV=0, MASTER_ADDR/PORT, etc.)
. "$PSScriptRoot\..\set_env.ps1"

# Activate venv if not already active.
if (-not $env:VIRTUAL_ENV) {
    & "$PSScriptRoot\..\.venv-win\Scripts\Activate.ps1"
}

# accelerate launch with c10d rendezvous (static rdzv backend hits the libuv bug
# on Windows even with USE_LIBUV=0 set, because torchrun pre-imports torch).
& "$PSScriptRoot\..\.venv-win\Scripts\accelerate.exe" launch `
    --multi_gpu --num_processes 4 --num_machines 1 `
    --mixed_precision bf16 `
    --main_process_port 29501 `
    --rdzv_backend c10d `
    "$PSScriptRoot\..\src\train_qwen_edit_lora\train_qwen_edit_lora.py" `
    --gradient_checkpointing `
    --batch_size 1 `
    --gradient_accumulation_steps 2 `
    --num_workers 0 `
    --epochs 1 `
    --resolution 768
