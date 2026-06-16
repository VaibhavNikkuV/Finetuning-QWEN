"""
Pre-download Qwen-Image-Edit-2511 model weights into $HF_HOME ahead of training,
so the actual training run isn't held hostage by a flaky network or HF outage.

Run:
    . .\set_env.ps1            # sets HF_HOME=E:\FinetuningQwen\hf_cache
    python scripts\precache_models.py

Set HF_TOKEN in the environment if any model is gated:
    $env:HF_TOKEN = "hf_..."   # or huggingface-cli login

Total download is ~15-18 GB on the first run. Subsequent runs are no-ops.
"""
from __future__ import annotations

import os
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


REPOS = [
    # The 4-bit quantized base we actually train against. Fits 24 GB A5000s.
    (
        "toandev/Qwen-Image-Edit-2511-4bit",
        ["transformer/*", "model_index.json", "scheduler/*", "processor/*"],
    ),
    # Text encoder + VAE from the official Qwen-Image repo (DiffSynth's recipe
    # mixes weights from Qwen-Image-Edit-2511 + Qwen-Image).
    (
        "Qwen/Qwen-Image",
        ["text_encoder/*", "vae/*", "tokenizer/*"],
    ),
    # Edit processor (image preprocessor) from the non-quantized Qwen-Image-Edit.
    (
        "Qwen/Qwen-Image-Edit",
        ["processor/*", "tokenizer*", "*.json"],
    ),
]


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        print("WARNING: HF_HOME is not set. Models will go to the user-default cache "
              "(typically C:\\Users\\<you>\\.cache\\huggingface). Source set_env.ps1 first "
              "if you want them on E:.", file=sys.stderr)
    else:
        print(f"HF_HOME = {hf_home}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Note: HF_TOKEN not set. Public models will work; gated models will fail.")

    failed: list[tuple[str, str]] = []
    for repo, patterns in REPOS:
        print(f"\n=== Downloading {repo} ({patterns}) ===")
        try:
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=patterns,
                token=token,
            )
            print(f"  -> {path}")
        except HfHubHTTPError as e:
            print(f"  FAILED ({e.response.status_code}): {e}", file=sys.stderr)
            failed.append((repo, str(e)))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append((repo, str(e)))

    if failed:
        print("\nSome downloads failed:", file=sys.stderr)
        for repo, msg in failed:
            print(f"  - {repo}: {msg}", file=sys.stderr)
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
