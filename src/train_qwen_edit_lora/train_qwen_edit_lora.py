#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 LoRA Fine-Tuning Script  (CORRECTED)
=========================================================
Fine-tunes Qwen-Image-Edit-2511 (4-bit base) with LoRA to add teacher
comments (red-ink annotations) to student answer sheets.

What was fixed vs the original (search for "[FIX]"):
  1. WSL2 NCCL crash (CUDA error 999): the old gloo fallback keyed on
     platform.system()=="Windows", but WSL reports "Linux", so it used NCCL
     and hit the WSL P2P failure. We now (a) keep NCCL on Linux/WSL but
     disable the transports WSL breaks, and (b) only use gloo on real Windows.
  2. Gradient checkpointing + frozen 4-bit base produced ZERO LoRA grads
     (silent no-op training). Added transformer.enable_input_require_grads().
  3. transformer(...) was missing the REQUIRED img_shapes and txt_seq_lens
     args -> wrong/!broken RoPE. Now built from the latent shape, for BOTH
     the noisy stream and the concatenated control stream.
  4. timestep was passed as t*1000 (range [0,1000]); the model wants the
     sigma in [0,1]. Fixed.
  5. LR-scheduler step math double-counted by num_processes (cosine never
     finished). Steps are now computed AFTER prepare() from the sharded
     dataloader, then the scheduler is prepared separately.
  6. DDP broadcast of the identical frozen 4-bit base could choke on the
     Params4bit subclass. broadcast_buffers=False added.
  7. Longer process-group timeout so the ~2 min shard load doesn't trip the
     default 10 min barrier and desync the ranks.

Multi-GPU launch (4xA5000):
    accelerate launch --num_processes 4 train_qwen_edit_lora.py --gradient_checkpointing
Single-GPU:
    python train_qwen_edit_lora.py --gradient_checkpointing
"""

import os
import platform

# [FIX 1] Set distributed env BEFORE torch / torch.distributed are touched.
# USE_LIBUV=0: Windows torch wheels lack libuv (harmless on Linux).
os.environ.setdefault("USE_LIBUV", "0")
# WSL2's NCCL P2P/SHM transports are unreliable and raise "Cuda failure 999".
# Disabling them forces NCCL onto a working path. No effect on native Linux.
if "microsoft" in platform.uname().release.lower():
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")

import math
import argparse
import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import (
    InitProcessGroupKwargs,
    DistributedDataParallelKwargs,
    set_seed,
)
from diffusers import QwenImageEditPlusPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

Image.MAX_IMAGE_PIXELS = 200_000_000  # ~14k x 14k scanned sheets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinned revision of the community 4-bit fork (protects against silent reweights).
DEFAULT_MODEL_REVISION = "bbf3075e1aee61f91027fa212c9a3a03cd5dc7c4"


# ============================================================================
# DATASET
# ============================================================================
class QwenEditDataset(Dataset):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, control_dir, target_dir, caption_dir, resolution=768):
        self.control_dir = Path(control_dir)
        self.target_dir = Path(target_dir)
        self.caption_dir = Path(caption_dir)
        self.resolution = resolution

        control_files = {f.stem: f for f in self.control_dir.iterdir()
                         if f.suffix.lower() in self.IMAGE_EXTENSIONS}
        target_files = {f.stem: f for f in self.target_dir.iterdir()
                        if f.suffix.lower() in self.IMAGE_EXTENSIONS}
        caption_files = {f.stem: f for f in self.caption_dir.iterdir()
                         if f.suffix.lower() == ".txt"}

        self.keys = sorted(set(control_files) & set(target_files) & set(caption_files))
        self.control_files = control_files
        self.target_files = target_files
        self.caption_files = caption_files
        logger.info(f"Found {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def _load_pil(self, p):
        # NOTE: forces a square. Answer sheets are portrait, so this distorts
        # aspect ratio. Square keeps batching trivial; if geometry matters,
        # switch to aspect-preserving resize+pad and use batch_size=1 (or
        # resolution bucketing), since non-square batches can't be stacked.
        return Image.open(p).convert("RGB").resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )

    def _to_tensor(self, img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

    def __getitem__(self, idx):
        key = self.keys[idx]
        control_pil = self._load_pil(self.control_files[key])
        target_pil = self._load_pil(self.target_files[key])
        return {
            "control": self._to_tensor(control_pil),
            "target": self._to_tensor(target_pil),
            "control_pil": control_pil,
            "caption": self.caption_files[key].read_text().strip(),
        }


def collate_fn(batch):
    return {
        "control": torch.stack([b["control"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "control_pil": [b["control_pil"] for b in batch],
        "caption": [b["caption"] for b in batch],
    }


# ============================================================================
# VAE / LATENT HELPERS
# ============================================================================
def encode_vae(vae, pixels, dtype):
    """Encode pixels with the QwenImage (WAN 2.1, video-style) VAE.

    Expects 5D [B, C, T, H, W]; for stills we add T=1 then squeeze it back.
    Normalization uses per-channel latents_mean/std from vae.config
    (NOT scaling_factor, which is the SD/SDXL convention).
    """
    pixels = pixels.to(device=vae.device, dtype=dtype)
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]

    latents = vae.encode(pixels).latent_dist.sample()

    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=latents.device,
                        dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device,
                       dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
    latents = (latents - mean) / std

    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.squeeze(2)  # back to [B,C,H,W]
    return latents


def pack_latents(latents):
    """[B,C,H,W] -> [B,(H/2)*(W/2),C*4]  (2x2 patchify; matches diffusers
    QwenImagePipeline._pack_latents)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def compute_loss(pred, noise, latents):
    # Flow-matching velocity target: d/dsigma[(1-sigma)x0 + sigma*eps] = eps - x0.
    target = noise - latents
    return F.mse_loss(pred.float(), target.float())


# ============================================================================
# CHECKPOINTING
# ============================================================================
def save_lora(accelerator, transformer, output_dir, step, final=False):
    if not accelerator.is_main_process:
        return
    suffix = "final" if final else f"step_{step}"
    save_path = Path(output_dir) / suffix
    save_path.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(transformer).save_pretrained(save_path)
    logger.info(f"Saved LoRA to {save_path}")


# ============================================================================
# TRAIN
# ============================================================================
def train(args):
    # [FIX 1] Backend: gloo ONLY on real Windows; NCCL on Linux AND WSL.
    # Under WSL the NCCL_P2P_DISABLE/SHM_DISABLE env (set at top) makes NCCL work.
    init_pg_backend = "gloo" if platform.system() == "Windows" else "nccl"

    # [FIX 6/7] broadcast_buffers=False avoids re-broadcasting the identical
    # frozen 4-bit base; longer timeout survives the slow shard load.
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    pg_kwargs = InitProcessGroupKwargs(
        backend=init_pg_backend, timeout=timedelta(minutes=30)
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[pg_kwargs, ddp_kwargs],
    )
    if args.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers("qwen-edit-lora", config=vars(args))

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    logger.info("Loading base pipeline...")
    pipeline_kwargs = {"torch_dtype": weight_dtype}
    if args.model_revision:
        pipeline_kwargs["revision"] = args.model_revision
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_name, **pipeline_kwargs)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
    )
    transformer = get_peft_model(pipe.transformer, lora_config)

    if args.gradient_checkpointing:
        try:
            transformer.gradient_checkpointing_enable()
            # [FIX 2] CRITICAL with a frozen base: without this, checkpointed
            # blocks recompute activations from inputs that don't require grad,
            # so NO gradient reaches the LoRA adapters and loss never moves.
            transformer.enable_input_require_grads()
            logger.info("Gradient checkpointing + input_require_grads enabled.")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    # Do NOT .to(device) the 4-bit transformer; bnb places it on the current
    # CUDA device (= local rank under accelerate launch) and prepare() wraps it.
    transformer.print_trainable_parameters()

    dataset = QwenEditDataset(args.control_dir, args.target_dir,
                              args.caption_dir, args.resolution)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # [FIX 5] Prepare model/optimizer/dataloader FIRST so len(dataloader) is the
    # per-process (sharded) length, then size the scheduler from that.
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    num_training_steps = num_update_steps_per_epoch * args.epochs

    # Accelerate steps a prepared scheduler num_processes times per optim step,
    # so multiply BOTH counts by num_processes (and use the sharded step count).
    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps * accelerator.num_processes,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info(f"Training: {num_training_steps} update steps "
                f"({num_update_steps_per_epoch}/epoch x {args.epochs} epochs)")
    global_step = 0

    for epoch in range(args.epochs):
        transformer.train()
        progress = tqdm(dataloader, disable=not accelerator.is_local_main_process,
                        desc=f"epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    target_latents = encode_vae(pipe.vae, batch["target"], weight_dtype)
                    control_latents = encode_vae(pipe.vae, batch["control"], weight_dtype)

                    # EditPlus encode_prompt wants prompt: List[str] and
                    # image: List[List[PIL]] (outer=batch, inner=ref images).
                    image_input = [[img] for img in batch["control_pil"]]
                    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                        prompt=batch["caption"],
                        image=image_input,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                    )

                bsz = target_latents.shape[0]

                # Flow-matching forward process. sigma == t in [0,1].
                t = torch.rand(bsz, device=target_latents.device, dtype=weight_dtype)
                sigmas = t.view(-1, 1, 1, 1)
                noise = torch.randn_like(target_latents)
                noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise

                # Pack and concat the control latents along the sequence dim.
                noisy_packed = pack_latents(noisy_latents)
                control_packed = pack_latents(control_latents)
                hidden_states = torch.cat([noisy_packed, control_packed], dim=1)

                # [FIX 3] Build img_shapes (one (1,h,w) per concatenated stream)
                # and txt_seq_lens from the actual latent grid. Both required.
                _, _, lh, lw = noisy_latents.shape
                ph, pw = lh // 2, lw // 2          # grid after 2x2 packing
                img_shapes = [[(1, ph, pw), (1, ph, pw)]] * bsz  # noisy + control
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                pred = transformer(
                    hidden_states=hidden_states,
                    timestep=t,                    # [FIX 4] sigma in [0,1], not t*1000
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,         # [FIX 3]
                    txt_seq_lens=txt_seq_lens,     # [FIX 3]
                    return_dict=False,
                )[0]

                # Keep only the noisy-stream tokens (drop the control tokens).
                target_seq_len = noisy_packed.shape[1]
                pred_target = pred[:, :target_seq_len]
                assert pred_target.shape == noisy_packed.shape, (
                    f"pred slice {tuple(pred_target.shape)} != noisy "
                    f"{tuple(noisy_packed.shape)} -- check img_shapes / concat order"
                )

                noise_packed = pack_latents(noise)
                target_packed = pack_latents(target_latents)
                loss = compute_loss(pred_target, noise_packed, target_packed)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_every == 0:
                    logs = {
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    }
                    progress.set_postfix(loss=f"{logs['train/loss']:.4f}",
                                         lr=f"{logs['train/lr']:.2e}")
                    if args.use_wandb:
                        accelerator.log(logs, step=global_step)

                if args.save_every and global_step % args.save_every == 0:
                    save_lora(accelerator, transformer, args.output_dir, global_step)

        logger.info(f"Epoch {epoch + 1} complete")

    save_lora(accelerator, transformer, args.output_dir, global_step, final=True)
    accelerator.end_training()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required on Windows; harmless on Linux

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _DATA = _REPO_ROOT / "qwen_edit_dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="toandev/Qwen-Image-Edit-2511-4bit")
    parser.add_argument("--model_revision", default=DEFAULT_MODEL_REVISION,
                        help="Pass '' to disable pinning.")
    parser.add_argument("--control_dir", default=str(_DATA / "control_1"))
    parser.add_argument("--target_dir",  default=str(_DATA / "targets"))
    parser.add_argument("--caption_dir", default=str(_DATA / "captions"))
    parser.add_argument("--output_dir",  default=str(_REPO_ROOT / "lora_output"))
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["to_k", "to_q", "to_v", "to_out.0"],
                        help="Image-stream attention only. Optionally add the "
                             "joint-attn text projections add_k_proj/add_q_proj/"
                             "add_v_proj/to_add_out for a stronger adapter.")
    parser.add_argument("--resolution", type=int, default=768,
                        help="768 fits 24 GB with 4-bit base + LoRA + grad ckpt. "
                             "Raise to 1024 only after a clean 768 run.")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)