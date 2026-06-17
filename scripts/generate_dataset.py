"""
Build the Qwen-Image-Edit training dataset from the raw CSVs in dataset/.

Each CSV row holds two image URLs (a clean student answer sheet and the same
sheet annotated with teacher comments) plus the comments themselves. This script
downloads both images and assembles the (control, target, caption) triplet layout
that the trainer (src/train_qwen_edit_lora) expects, split into one self-contained
dataset per language.

Output:
    qwen_edit_dataset/
      hindi/
        control_1/<key>.png   (original_img_link  - clean input)
        targets/<key>.png     (generated_img_link - annotated output)
        captions/<key>.txt    (prompt.txt filled with comments + language)
        dataset.json          (manifest of completed triplets)
      english/
        control_1/  targets/  captions/  dataset.json

    where <key> = "<copy_id>_p<page_no>" (same stem across all three folders, as
    QwenEditDataset matches samples by stem intersection).

Run:
    python scripts/generate_dataset.py                 # full build (10 entries in parallel)
    python scripts/generate_dataset.py --workers 20    # tune how many entries run at once
    python scripts/generate_dataset.py --limit 2       # quick test (2 rows/lang)
    python scripts/generate_dataset.py --overwrite     # re-download everything
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Language -> CSV filename (both live under --dataset-dir).
LANGUAGE_CSVS = {
    "hindi": "hindi_image_dataset.csv",
    "english": "english_image_dataset.csv",
}

USER_AGENT = "qwen-edit-dataset-builder/1.0"


def build_caption(template: str, comments: list[str], language: str) -> str:
    """Fill the prompt template's {comments} (bulleted) and {language} slots.

    Uses str.replace rather than str.format so the template's prose (which may
    contain stray braces) is left untouched.
    """
    bullets = "\n".join(f"- {c}" for c in comments) if comments else "- (none)"
    return template.replace("{comments}", bullets).replace("{language}", language)


def download(url: str, dest: Path, *, timeout: float, retries: int,
             overwrite: bool) -> None:
    """Download url to dest with retries. Skips if dest exists (unless overwrite).

    Writes to a .tmp sibling then atomically renames, so an interrupted download
    never leaves a file that looks complete.
    """
    if dest.exists() and not overwrite:
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("empty response body")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                TimeoutError, OSError) as err:
            last_err = err
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))  # exponential backoff, capped
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} -> {dest.name}: {last_err}")


def parse_rows(csv_path: Path, limit: int) -> list[dict]:
    """Read a language CSV (UTF-8 BOM) into a list of validated row dicts."""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for i, raw in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            try:
                comments = json.loads(raw["on_page_comments"])
                if not isinstance(comments, list):
                    raise ValueError("on_page_comments is not a JSON list")
            except (ValueError, KeyError) as err:
                print(f"  ! skipping malformed row {i} in {csv_path.name}: {err}",
                      file=sys.stderr)
                continue
            rows.append({
                "key": f"{raw['copy_id']}_p{raw['page_no']}",
                "copy_id": raw["copy_id"],
                "page_no": raw["page_no"],
                "language": raw["language"],
                "control_url": raw["original_img_link"],
                "target_url": raw["generated_img_link"],
                "comments": [str(c) for c in comments],
            })
    return rows


def process_language(language: str, csv_path: Path, template: str, out_root: Path,
                     args: argparse.Namespace) -> None:
    print(f"\n=== {language} ({csv_path.name}) ===")
    rows = parse_rows(csv_path, args.limit)
    if not rows:
        print("  no usable rows; skipping.")
        return

    lang_dir = out_root / language
    control_dir = lang_dir / "control_1"
    target_dir = lang_dir / "targets"
    caption_dir = lang_dir / "captions"
    for d in (control_dir, target_dir, caption_dir):
        d.mkdir(parents=True, exist_ok=True)

    def fetch_pair(row: dict) -> tuple[dict, str | None]:
        """Download both images for a row. Returns (row, error-or-None)."""
        key = row["key"]
        try:
            download(row["control_url"], control_dir / f"{key}.png",
                     timeout=args.timeout, retries=args.retries,
                     overwrite=args.overwrite)
            download(row["target_url"], target_dir / f"{key}.png",
                     timeout=args.timeout, retries=args.retries,
                     overwrite=args.overwrite)
            return row, None
        except Exception as err:  # noqa: BLE001 - record and continue
            return row, str(err)

    manifest: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_pair, row) for row in rows]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{language} images", unit="pair"):
            row, err = fut.result()
            key = row["key"]
            if err is not None:
                failures.append(f"{key}\t{err}")
                continue

            # Both images present -> safe to emit the caption + manifest entry.
            caption = build_caption(template, row["comments"], row["language"])
            (caption_dir / f"{key}.txt").write_text(caption, encoding="utf-8")
            manifest.append({
                "key": key,
                "control_1": f"control_1/{key}.png",
                "target": f"targets/{key}.png",
                "caption": f"captions/{key}.txt",
                "language": row["language"],
                "copy_id": row["copy_id"],
                "page_no": row["page_no"],
            })

    manifest.sort(key=lambda e: e["key"])
    (lang_dir / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"  wrote {len(manifest)} triplets, skipped {len(failures)} "
          f"-> {lang_dir}")
    if failures:
        log = lang_dir / "failures.log"
        log.write_text("\n".join(failures), encoding="utf-8")
        print(f"  failures logged to {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"),
                        help="Directory holding the CSVs and prompt.txt.")
    parser.add_argument("--out", type=Path, default=Path("qwen_edit_dataset"),
                        help="Output dataset root.")
    parser.add_argument("--workers", type=int, default=50,
                        help="Number of entries (rows) processed in parallel. "
                             "Change this to tune concurrency.") 
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-request download timeout (seconds).")
    parser.add_argument("--retries", type=int, default=3,
                        help="Download attempts per image before giving up.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download images even if already present.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap rows per language (0 = all); for quick tests.")
    args = parser.parse_args()

    template = (args.dataset_dir / "prompt.txt").read_text(encoding="utf-8")

    for language, csv_name in LANGUAGE_CSVS.items():
        csv_path = args.dataset_dir / csv_name
        if not csv_path.exists():
            print(f"! {csv_path} not found; skipping {language}.", file=sys.stderr)
            continue
        process_language(language, csv_path, template, args.out, args)

    print(f"\nDone. Dataset root: {args.out.resolve()}")


if __name__ == "__main__":
    main()
