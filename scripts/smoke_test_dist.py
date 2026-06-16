"""
Standalone multi-process gloo init smoke test - no model, no data, no DiffSynth.
Just verifies that 4 ranks can come up under accelerate launch and exit clean.

Run via:
    . .\set_env.ps1
    .\.venv-win\Scripts\accelerate.exe launch `
        --multi_gpu --num_processes 4 --num_machines 1 `
        --mixed_precision bf16 --main_process_port 29500 `
        scripts\smoke_test_dist.py
"""
import os
import torch
import torch.distributed as dist


def main() -> None:
    backend = os.environ.get("ACCELERATE_TORCH_DISTRIBUTED_BACKEND", "gloo")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()

    # All-reduce a tiny tensor so we exercise the actual collective path,
    # not just init/destroy.
    t = torch.tensor([rank], dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(world))

    print(f"[rank {rank}/{world}] backend={backend} all_reduce_ok={int(t.item()) == expected}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
