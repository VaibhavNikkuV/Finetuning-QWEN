# Source this from PowerShell before running training:  . .\set_env.ps1
# Sets HF cache, distributed backend, and gloo-on-Windows quirks.

$env:HF_HOME = "E:\FinetuningQwen\hf_cache"

# Force gloo backend - NCCL has no Windows build (PyTorch 2.4.1+cu124 ships gloo only).
$env:ACCELERATE_TORCH_DISTRIBUTED_BACKEND = "gloo"
$env:TORCH_DISTRIBUTED_DEFAULT_BACKEND = "gloo"

# OMP threads per rank: cores / num_processes (tune per box; 8 is a sane default for 4 ranks on a 32-core).
$env:OMP_NUM_THREADS = "8"

# TCP rendezvous - 127.0.0.1 (NOT "localhost", which can resolve to ::1 on Win11; gloo binds v4).
$env:MASTER_ADDR = "127.0.0.1"
$env:MASTER_PORT = "29500"

# GLOO_SOCKET_IFNAME on Windows: leave UNSET. Gloo's UV transport doesn't recognize
# "Loopback" as a Windows iface name (fails with "Unable to find address for:
# Loopback"). Letting it auto-pick uses the default v4 iface, which is what we want
# on a single-host setup.

# Windows torch wheels are built without libuv; torchrun/accelerate default to libuv
# for TCPStore and crash with "use_libuv was requested but PyTorch was build without
# libuv support". Disable.
$env:USE_LIBUV = "0"

# Point accelerate at the project-local config we wrote.
$env:ACCELERATE_CONFIG_FILE = (Join-Path (Get-Location) "accelerate_config.yaml")

Write-Host "env: HF_HOME=$env:HF_HOME"
Write-Host "env: ACCELERATE_TORCH_DISTRIBUTED_BACKEND=$env:ACCELERATE_TORCH_DISTRIBUTED_BACKEND"
Write-Host "env: MASTER_ADDR=$env:MASTER_ADDR  MASTER_PORT=$env:MASTER_PORT"
Write-Host "env: ACCELERATE_CONFIG_FILE=$env:ACCELERATE_CONFIG_FILE"
