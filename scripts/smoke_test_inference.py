"""
Single-GPU inference smoke test. Loads the toandev 4-bit Qwen-Image-Edit-2511
pipeline (the same path the legacy trainer uses), runs ONE edit step on a
white control image, and saves the result. If this passes, training will at
least *load*.

Run:
    . .\set_env.ps1
    $env:CUDA_VISIBLE_DEVICES = "0"
    python scripts\smoke_test_inference.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="toandev/Qwen-Image-Edit-2511-4bit")
    parser.add_argument("--revision", default="bbf3075e1aee61f91027fa212c9a3a03cd5dc7c4")
    parser.add_argument("--out", default="smoke_test_inference.png")
    parser.add_argument("--prompt", default="Add a small red mark in the corner.")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps. 4 is enough to verify pipeline runs; "
                             "real generation needs 30+.")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    print(f"HF_HOME = {os.environ.get('HF_HOME', '<unset>')}")
    print(f"CUDA available = {torch.cuda.is_available()}, "
          f"device_count = {torch.cuda.device_count()}")

    print(f"\nLoading {args.model_name} @ rev {args.revision[:8]}...")
    t0 = time.time()
    # Don't call pipe.to('cuda') after loading: bnb-quantized weights are
    # already on GPU, and accelerate <1.0 raises on .to() for quantized
    # pipelines. Move only the non-quantized components (vae, text_encoder)
    # explicitly below.
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_name,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    pipe.vae.to("cuda")
    pipe.text_encoder.to("cuda")
    print(f"  vae/text_encoder moved to cuda; transformer stays bnb-quantized "
          f"on its existing device")

    control = Image.new("RGB", (args.size, args.size), "white")

    print(f"\nRunning {args.steps}-step edit at {args.size}x{args.size} ...")
    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        image=[control],
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        true_cfg_scale=1.0,
    )
    print(f"  inference in {time.time() - t0:.1f}s")

    img = out.images[0] if hasattr(out, "images") else out
    out_path = Path(args.out).resolve()
    img.save(out_path)
    print(f"\nWrote {out_path}")
    print(f"VRAM peak this run: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
