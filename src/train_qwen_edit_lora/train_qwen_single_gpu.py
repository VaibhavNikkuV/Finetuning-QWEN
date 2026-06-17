#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 LoRA fine-tuning — SINGLE GPU (simple version).

No accelerate, no DDP, no process groups. One GPU, one process. Keeps the
verified-correct training internals: img_shapes/txt_seq_lens in the forward,
sigma-in-[0,1] timestep, flow-matching velocity target, and the
gradient-checkpointing grad-flow fix.

Run:
    python train_qwen_edit_lora_single_gpu.py --gradient_checkpointing

Dataset layout (file stems must match across the three folders):
    qwen_edit_dataset/
        control_1/   student (input) images
        targets/     teacher-annotated (output) images
        captions/    .txt editing instructions
"""

import os
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import QwenImageEditPlusPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

Image.MAX_IMAGE_PIXELS = 200_000_000  # ~14k x 14k scanned sheets
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_REVISION = "bbf3075e1aee61f91027fa212c9a3a03cd5dc7c4"


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class QwenEditDataset(Dataset):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, control_dir, target_dir, caption_dir, resolution=768):
        self.control_dir = Path(control_dir)
        self.target_dir = Path(target_dir)
        self.caption_dir = Path(caption_dir)
        self.resolution = resolution

        control = {f.stem: f for f in self.control_dir.iterdir()
                   if f.suffix.lower() in self.IMAGE_EXTENSIONS}
        target = {f.stem: f for f in self.target_dir.iterdir()
                  if f.suffix.lower() in self.IMAGE_EXTENSIONS}
        caption = {f.stem: f for f in self.caption_dir.iterdir()
                   if f.suffix.lower() == ".txt"}

        self.keys = sorted(set(control) & set(target) & set(caption))
        self.control, self.target, self.caption = control, target, caption
        if not self.keys:
            raise RuntimeError("No matching (control, target, caption) stems found.")
        logger.info(f"Found {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def _load_pil(self, p):
        # Square resize keeps batching simple; distorts portrait sheets.
        return Image.open(p).convert("RGB").resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )

    def _to_tensor(self, img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

    def __getitem__(self, idx):
        k = self.keys[idx]
        control_pil = self._load_pil(self.control[k])
        target_pil = self._load_pil(self.target[k])
        return {
            "control": self._to_tensor(control_pil),
            "target": self._to_tensor(target_pil),
            "control_pil": control_pil,
            "caption": self.caption[k].read_text().strip(),
        }


def collate_fn(batch):
    return {
        "control": torch.stack([b["control"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "control_pil": [b["control_pil"] for b in batch],
        "caption": [b["caption"] for b in batch],
    }


# ---------------------------------------------------------------------------
# VAE / LATENT HELPERS
# ---------------------------------------------------------------------------
def encode_vae(vae, pixels, dtype):
    """QwenImage VAE is video-style: expects 5D [B,C,T,H,W]; add T=1 for stills.
    Normalize with per-channel latents_mean/std (NOT scaling_factor)."""
    pixels = pixels.to(device=vae.device, dtype=dtype)
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(2)
    latents = vae.encode(pixels).latent_dist.sample()
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=latents.device,
                        dtype=latents.dtype).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device,
                       dtype=latents.dtype).view(1, z, 1, 1, 1)
    latents = (latents - mean) / std
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.squeeze(2)
    return latents


def pack_latents(latents):
    """[B,C,H,W] -> [B,(H/2)*(W/2),C*4] (matches diffusers _pack_latents)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def compute_loss(pred, noise, latents):
    target = noise - latents  # flow-matching velocity: eps - x0
    return F.mse_loss(pred.float(), target.float())


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible.")
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    logger.info("Loading base pipeline (4-bit)...")
    pipe_kwargs = {"torch_dtype": weight_dtype}
    if args.model_revision:
        pipe_kwargs["revision"] = args.model_revision
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_name, **pipe_kwargs)

    # Freeze everything except the LoRA adapters we add below.
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.to(device)
    pipe.text_encoder.to(device)
    # The 4-bit transformer is already placed on the GPU by bitsandbytes.

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
    )
    transformer = get_peft_model(pipe.transformer, lora_config)

    if args.gradient_checkpointing:
        # diffusers models use enable_gradient_checkpointing() -- NOT the
        # transformers-style gradient_checkpointing_enable(). This resolves
        # through the PEFT wrapper down to the QwenImageTransformer2DModel.
        # (Do NOT call enable_input_require_grads(): diffusers transformers have
        #  no get_input_embeddings(), so it raises NotImplementedError. We make
        #  the input require grad in the loop instead.)
        transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")

    transformer.print_trainable_parameters()
    transformer.train()

    dataset = QwenEditDataset(args.control_dir, args.target_dir,
                              args.caption_dir, args.resolution)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.epochs
    lr_scheduler = get_scheduler(
        args.scheduler, optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(f"Training {num_training_steps} steps "
                f"({steps_per_epoch}/epoch x {args.epochs} epochs)")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(progress):
            # ---- encode inputs (no grad) ----
            with torch.no_grad():
                target_latents = encode_vae(pipe.vae, batch["target"], weight_dtype)
                control_latents = encode_vae(pipe.vae, batch["control"], weight_dtype)
                image_input = [[img] for img in batch["control_pil"]]
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                    prompt=batch["caption"],
                    image=image_input,
                    device=device,
                    num_images_per_prompt=1,
                )
                # encode_prompt returns mask=None when it is all-ones (no
                # padding), which always happens at batch_size=1. The
                # transformer accepts None, but we need a concrete mask to
                # compute txt_seq_lens, so rebuild an all-ones mask.
                if prompt_embeds_mask is None:
                    prompt_embeds_mask = torch.ones(
                        prompt_embeds.shape[:2], dtype=torch.long,
                        device=prompt_embeds.device,
                    )

            bsz = target_latents.shape[0]

            # ---- flow-matching forward process (sigma == t in [0,1]) ----
            t = torch.rand(bsz, device=device, dtype=weight_dtype)
            sigmas = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(target_latents)
            noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise

            noisy_packed = pack_latents(noisy_latents)
            control_packed = pack_latents(control_latents)
            hidden_states = torch.cat([noisy_packed, control_packed], dim=1)
            if args.gradient_checkpointing:
                # Frozen base + checkpointing: without an input that requires
                # grad, torch warns "None of the inputs have requires_grad=True"
                # and LoRA grads come back None. This is the diffusers-friendly
                # replacement for enable_input_require_grads().
                hidden_states.requires_grad_(True)

            _, _, lh, lw = noisy_latents.shape
            ph, pw = lh // 2, lw // 2
            img_shapes = [[(1, ph, pw), (1, ph, pw)]] * bsz  # noisy + control
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

            # ---- denoise ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = transformer(
                    hidden_states=hidden_states,
                    timestep=t,                       # sigma in [0,1]
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

            target_seq_len = noisy_packed.shape[1]
            pred_target = pred[:, :target_seq_len]
            assert pred_target.shape == noisy_packed.shape, (
                f"pred slice {tuple(pred_target.shape)} != noisy "
                f"{tuple(noisy_packed.shape)} -- check img_shapes / concat order"
            )

            loss = compute_loss(pred_target, pack_latents(noise),
                                pack_latents(target_latents))
            (loss / args.gradient_accumulation_steps).backward()

            # ---- optimizer step (with manual grad accumulation) ----
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )
                if args.save_every and global_step % args.save_every == 0:
                    p = out_dir / f"step_{global_step}"
                    transformer.save_pretrained(p)
                    logger.info(f"Saved LoRA to {p}")

        logger.info(f"Epoch {epoch + 1} complete")

    transformer.save_pretrained(out_dir / "final")
    logger.info(f"Saved final LoRA to {out_dir / 'final'}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _DATA = _REPO_ROOT / "qwen_edit_dataset"

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="toandev/Qwen-Image-Edit-2511-4bit")
    p.add_argument("--model_revision", default=DEFAULT_MODEL_REVISION,
                   help="Pass '' to disable revision pinning.")
    p.add_argument("--control_dir", default=str(_DATA / "control_1"))
    p.add_argument("--target_dir",  default=str(_DATA / "targets"))
    p.add_argument("--caption_dir", default=str(_DATA / "captions"))
    p.add_argument("--output_dir",  default=str(_REPO_ROOT / "lora_output"))
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+",
                   default=["to_k", "to_q", "to_v", "to_out.0"])
    p.add_argument("--resolution", type=int, default=768,
                   help="768 fits 24 GB with 4-bit base + LoRA + grad ckpt.")
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", default="bf16")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()
    train(args)