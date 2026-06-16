"""
Generate a tiny synthetic dataset for SMOKE-TESTING the training pipeline.
NOT for real training - the data is meaningless visually, just structured to
match the trainer's (control, target, caption) triplet expectation.

Output:
    qwen_edit_dataset_smoke/
      control_1/<id>.png    (768x768 white image)
      targets/<id>.png      (same image with a red rectangle overlay)
      captions/<id>.txt     ("Add a red rectangle.")

Run:
    python scripts\build_synthetic_micro_dataset.py [--n 8] [--size 768]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of (control, target, caption) triples to generate.")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--out", type=Path, default=Path("qwen_edit_dataset_smoke"))
    args = parser.parse_args()

    random.seed(0)

    ctrl_dir = args.out / "control_1"
    tgt_dir  = args.out / "targets"
    cap_dir  = args.out / "captions"
    for d in (ctrl_dir, tgt_dir, cap_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        key = f"sample_{i:03d}"

        control = Image.new("RGB", (args.size, args.size), "white")
        control.save(ctrl_dir / f"{key}.png")

        target = control.copy()
        draw = ImageDraw.Draw(target)
        x = random.randint(50, args.size - 200)
        y = random.randint(50, args.size - 100)
        draw.rectangle([x, y, x + 150, y + 60], outline="red", width=4)
        target.save(tgt_dir / f"{key}.png")

        (cap_dir / f"{key}.txt").write_text(
            "Add a red rectangle correction mark.", encoding="utf-8"
        )

    print(f"Wrote {args.n} triples to {args.out.resolve()}")


if __name__ == "__main__":
    main()
