# #!/usr/bin/env python3
# """
# Qwen-Image-Edit-2511 LoRA Fine-Tuning Script
# =============================================
# Fine-tunes the Qwen-Image-Edit-2511 model using LoRA to learn
# adding teacher comments (red ink annotations) to student answer sheets.

# Dataset structure (from generate_dataset.py):
#     qwen_edit_dataset/
#     ├── targets/       (teacher images — annotated output)
#     ├── control_1/     (student images — clean input)
#     └── captions/      (editing instructions from comments)

# Usage:
#     accelerate launch train_qwen_edit_lora.py --config train_config.yaml

#     # Single GPU:
#     python train_qwen_edit_lora.py --config train_config.yaml

# Requirements:
#     pip install torch torchvision accelerate diffusers transformers peft
#     pip install bitsandbytes omegaconf pillow tqdm wandb
#     pip install git+https://github.com/huggingface/diffusers
# """

# import argparse
# import gc
# import logging
# import math
# import os
# import random
# import sys
# from pathlib import Path
# from typing import Optional

# os.environ["USE_LIBUV"] = "0"

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from omegaconf import OmegaConf
# from tqdm import tqdm

# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration, set_seed

# from diffusers import (
#     QwenImageEditPlusPipeline,
#     AutoencoderKLQwenImage,
#     FlowMatchEulerDiscreteScheduler,
# )
# from diffusers.utils import convert_unet_state_dict_to_peft
# from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# from transformers import AutoProcessor

# logger = get_logger(__name__, log_level="INFO")


# # ============================================================================
# # DATASET
# # ============================================================================

# class QwenEditDataset(Dataset):
#     """
#     Dataset for Qwen-Image-Edit fine-tuning.

#     Loads triplets of (control_image, target_image, caption)
#     matched by filename stem.
#     """

#     IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

#     def __init__(
#         self,
#         img_dir: str,
#         control_dir: str,
#         caption_dir: str,
#         resolution: int = 1024,
#         center_crop: bool = True,
#     ):
#         self.img_dir = Path(img_dir)
#         self.control_dir = Path(control_dir)
#         self.caption_dir = Path(caption_dir)
#         self.resolution = resolution
#         self.center_crop = center_crop

#         # Discover matching triplets by filename stem
#         target_stems = {
#             f.stem: f for f in self.img_dir.iterdir()
#             if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
#         }
#         control_stems = {
#             f.stem: f for f in self.control_dir.iterdir()
#             if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
#         }
#         caption_stems = {
#             f.stem: f for f in self.caption_dir.iterdir()
#             if f.is_file() and f.suffix.lower() == ".txt"
#         }

#         common = set(target_stems.keys()) & set(control_stems.keys()) & set(caption_stems.keys())
#         self.samples = sorted([
#             {
#                 "stem": s,
#                 "target": target_stems[s],
#                 "control": control_stems[s],
#                 "caption": caption_stems[s],
#             }
#             for s in common
#         ], key=lambda x: x["stem"])

#         logger.info(f"Dataset: {len(self.samples)} samples found")
#         if not self.samples:
#             raise ValueError(
#                 f"No matching triplets found in:\n"
#                 f"  targets:  {self.img_dir}\n"
#                 f"  controls: {self.control_dir}\n"
#                 f"  captions: {self.caption_dir}"
#             )

#     def __len__(self):
#         return len(self.samples)

#     def _load_and_resize(self, path: Path) -> Image.Image:
#         """Load image and resize to target resolution."""
#         img = Image.open(path).convert("RGB")
#         w, h = img.size

#         if self.center_crop:
#             # Center crop to square, then resize
#             crop_size = min(w, h)
#             left = (w - crop_size) // 2
#             top = (h - crop_size) // 2
#             img = img.crop((left, top, left + crop_size, top + crop_size))

#         img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
#         return img

#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         target_img = self._load_and_resize(sample["target"])
#         control_img = self._load_and_resize(sample["control"])

#         caption = sample["caption"].read_text(encoding="utf-8").strip()

#         # Convert to tensors: [0, 255] uint8 → [-1, 1] float
#         target_tensor = torch.from_numpy(
#             __import__("numpy").array(target_img)
#         ).permute(2, 0, 1).float() / 127.5 - 1.0

#         control_tensor = torch.from_numpy(
#             __import__("numpy").array(control_img)
#         ).permute(2, 0, 1).float() / 127.5 - 1.0

#         return {
#             "target": target_tensor,
#             "control": control_tensor,
#             "caption": caption,
#             "stem": sample["stem"],
#         }


# # ============================================================================
# # HELPERS
# # ============================================================================

# def compute_flow_matching_loss(
#     model_output: torch.Tensor,
#     noise: torch.Tensor,
#     latents: torch.Tensor,
#     timesteps: torch.Tensor,
#     sigmas: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Compute flow matching loss for rectified flow models.
#     The target is: noise - latents (the velocity field).
#     """
#     target = noise - latents
#     loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
#     return loss


# def encode_images(vae, images, weight_dtype):
#     """Encode images to latent space using VAE."""
#     images = images.to(device=vae.device, dtype=weight_dtype)
#     latents = vae.encode(images).latent_dist.sample()
#     latents = latents * vae.config.scaling_factor
#     return latents


# def log_validation(
#     pipeline, accelerator, config, epoch, step, control_images, captions
# ):
#     """Generate validation images during training."""
#     logger.info(f"Running validation at step {step}...")

#     pipeline.to(accelerator.device)
#     generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

#     images = []
#     for ctrl_img, caption in zip(control_images, captions):
#         with torch.inference_mode():
#             output = pipeline(
#                 image=[ctrl_img],
#                 prompt=caption,
#                 negative_prompt=" ",
#                 num_inference_steps=20,
#                 true_cfg_scale=4.0,
#                 guidance_scale=1.0,
#                 generator=generator,
#             )
#             images.append(output.images[0])

#     # Save validation images
#     val_dir = Path(config.output_dir) / "validation" / f"step_{step}"
#     val_dir.mkdir(parents=True, exist_ok=True)

#     for i, (img, caption) in enumerate(zip(images, captions)):
#         img.save(val_dir / f"val_{i}.png")
#         (val_dir / f"val_{i}.txt").write_text(caption)

#     logger.info(f"Validation images saved to {val_dir}")

#     # Log to wandb if available
#     if accelerator.is_main_process:
#         try:
#             import wandb
#             if wandb.run is not None:
#                 wandb.log({
#                     f"validation/image_{i}": wandb.Image(img, caption=caption[:100])
#                     for i, (img, caption) in enumerate(zip(images, captions))
#                 }, step=step)
#         except ImportError:
#             pass

#     del images
#     torch.cuda.empty_cache()


# # ============================================================================
# # MAIN TRAINING FUNCTION
# # ============================================================================

# def train(config):
#     """Main training loop."""

#     # --- Accelerator Setup ---
#     project_config = ProjectConfiguration(
#         project_dir=config.output_dir,
#         logging_dir=config.get("logging_dir", os.path.join(config.output_dir, "logs")),
#     )

#     accelerator = Accelerator(
#         gradient_accumulation_steps=config.gradient_accumulation_steps,
#         mixed_precision=config.get("mixed_precision", "bf16"),
#         log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
#         project_config=project_config,
#     )

#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger.info(accelerator.state, main_process_only=False)

#     if config.get("seed"):
#         set_seed(config.seed)

#     weight_dtype = torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float32

#     # --- Load Model Components ---
#     logger.info("=" * 60)
#     logger.info("  Loading Qwen-Image-Edit-2511 components")
#     logger.info("=" * 60)

#     pipeline = QwenImageEditPlusPipeline.from_pretrained(
#         config.pretrained_model_name_or_path,
#         torch_dtype=weight_dtype,
#     )

#     # Extract components
#     vae = pipeline.vae
#     transformer = pipeline.transformer
#     text_encoder = pipeline.text_encoder
#     tokenizer = pipeline.tokenizer
#     scheduler = pipeline.scheduler

#     # Freeze everything except LoRA
#     vae.requires_grad_(False)
#     transformer.requires_grad_(False)
#     text_encoder.requires_grad_(False)

#     vae.to(accelerator.device, dtype=weight_dtype)
#     text_encoder.to(accelerator.device, dtype=weight_dtype)

#     # --- Apply LoRA ---
#     logger.info("Applying LoRA to transformer...")

#     target_modules = config.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"])
#     if isinstance(target_modules, list):
#         target_modules = list(target_modules)

#     lora_config = LoraConfig(
#         r=config.get("lora_rank", 32),
#         lora_alpha=config.get("lora_alpha", 64),
#         lora_dropout=config.get("lora_dropout", 0.05),
#         target_modules=target_modules,
#         init_lora_weights="gaussian",
#     )

#     transformer = get_peft_model(transformer, lora_config)
#     transformer.print_trainable_parameters()
#     transformer.to(accelerator.device)

#     if config.get("gradient_checkpointing", False):
#         transformer.enable_input_require_grads()
#         transformer.gradient_checkpointing_enable()

#     # --- Dataset ---
#     logger.info("Loading dataset...")
#     dataset = QwenEditDataset(
#         img_dir=config.img_dir,
#         control_dir=config.control_dir,
#         caption_dir=config.get("caption_dir", os.path.join(config.dataset_dir, "captions")),
#         resolution=config.get("resolution", 1024),
#         center_crop=config.get("center_crop", True),
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=config.get("train_batch_size", 1),
#         shuffle=True,
#         num_workers=config.get("dataloader_num_workers", 4),
#         pin_memory=True,
#         drop_last=True,
#     )

#     # --- Optimizer ---
#     if config.get("use_8bit_adam", False):
#         try:
#             import bitsandbytes as bnb
#             optimizer_cls = bnb.optim.AdamW8bit
#             logger.info("Using 8-bit AdamW optimizer")
#         except ImportError:
#             logger.warning("bitsandbytes not available, falling back to AdamW")
#             optimizer_cls = torch.optim.AdamW
#     else:
#         optimizer_cls = torch.optim.AdamW

#     trainable_params = [p for p in transformer.parameters() if p.requires_grad]
#     optimizer = optimizer_cls(
#         trainable_params,
#         lr=config.learning_rate,
#         betas=(0.9, 0.999),
#         weight_decay=1e-2,
#         eps=1e-8,
#     )

#     # --- LR Scheduler ---
#     num_training_steps = config.get("max_train_steps") or (
#         len(dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
#     )

#     from diffusers.optimization import get_scheduler
#     lr_scheduler = get_scheduler(
#         config.get("lr_scheduler", "cosine"),
#         optimizer=optimizer,
#         num_warmup_steps=config.get("lr_warmup_steps", 100) * accelerator.num_processes,
#         num_training_steps=num_training_steps * accelerator.num_processes,
#     )

#     # --- Prepare with Accelerator ---
#     transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
#         transformer, optimizer, dataloader, lr_scheduler
#     )

#     # --- Validation Setup ---
#     val_control_images = []
#     val_captions = []
#     num_val = config.get("num_validation_images", 2)
#     if num_val > 0 and len(dataset) > 0:
#         for i in range(min(num_val, len(dataset))):
#             sample = dataset[i]
#             # Convert back to PIL for validation pipeline
#             ctrl_pil = Image.fromarray(
#                 ((sample["control"].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
#             )
#             val_control_images.append(ctrl_pil)
#             val_captions.append(sample["caption"])

#     # --- Training ---
#     logger.info("=" * 60)
#     logger.info("  TRAINING START")
#     logger.info("=" * 60)
#     logger.info(f"  Num samples       = {len(dataset)}")
#     logger.info(f"  Num epochs        = {config.num_train_epochs}")
#     logger.info(f"  Batch size        = {config.train_batch_size}")
#     logger.info(f"  Grad accum steps  = {config.gradient_accumulation_steps}")
#     logger.info(f"  Effective batch   = {config.train_batch_size * config.gradient_accumulation_steps}")
#     logger.info(f"  Total steps       = {num_training_steps}")
#     logger.info(f"  Learning rate     = {config.learning_rate}")
#     logger.info(f"  LoRA rank         = {config.lora_rank}")
#     logger.info(f"  Resolution        = {config.resolution}")
#     logger.info("=" * 60)

#     global_step = 0
#     best_loss = float("inf")

#     for epoch in range(config.num_train_epochs):
#         transformer.train()
#         epoch_loss = 0.0

#         progress_bar = tqdm(
#             dataloader,
#             desc=f"Epoch {epoch + 1}/{config.num_train_epochs}",
#             disable=not accelerator.is_local_main_process,
#         )

#         for step, batch in enumerate(progress_bar):
#             with accelerator.accumulate(transformer):
#                 # --- Encode target and control images to latents ---
#                 with torch.no_grad():
#                     target_latents = encode_images(vae, batch["target"], weight_dtype)
#                     control_latents = encode_images(vae, batch["control"], weight_dtype)

#                 # --- Encode text ---
#                 with torch.no_grad():
#                     text_inputs = tokenizer(
#                         batch["caption"],
#                         padding="max_length",
#                         max_length=tokenizer.model_max_length,
#                         truncation=True,
#                         return_tensors="pt",
#                     ).to(accelerator.device)

#                     encoder_output = text_encoder(
#                         text_inputs.input_ids,
#                         attention_mask=text_inputs.attention_mask,
#                     )
#                     prompt_embeds = encoder_output[0].to(dtype=weight_dtype)

#                 # --- Sample noise and timesteps (flow matching) ---
#                 bsz = target_latents.shape[0]
#                 noise = torch.randn_like(target_latents)

#                 # Uniform timestep sampling for flow matching
#                 timesteps = torch.rand(bsz, device=target_latents.device)
#                 sigmas = timesteps.view(-1, 1, 1, 1)

#                 # Interpolate: noisy_latents = (1 - sigma) * latents + sigma * noise
#                 noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise

#                 # --- Concatenate control latents with noisy target ---
#                 # For image editing, the model conditions on control + noisy target
#                 model_input = torch.cat([noisy_latents, control_latents], dim=1)

#                 # --- Forward pass through transformer ---
#                 model_output = transformer(
#                     hidden_states=model_input,
#                     timestep=timesteps * 1000,  # Scale to scheduler range
#                     encoder_hidden_states=prompt_embeds,
#                     return_dict=False,
#                 )[0]

#                 # --- Compute loss (flow matching: predict velocity) ---
#                 loss = compute_flow_matching_loss(
#                     model_output, noise, target_latents, timesteps, sigmas
#                 )

#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     accelerator.clip_grad_norm_(trainable_params, config.get("max_grad_norm", 1.0))

#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#             # --- Logging ---
#             if accelerator.sync_gradients:
#                 global_step += 1
#                 epoch_loss += loss.detach().item()

#                 if global_step % config.get("log_every_n_steps", 10) == 0:
#                     avg_loss = epoch_loss / global_step if global_step > 0 else loss.item()
#                     progress_bar.set_postfix({
#                         "loss": f"{loss.item():.4f}",
#                         "avg_loss": f"{avg_loss:.4f}",
#                         "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
#                     })

#                     if accelerator.is_main_process:
#                         try:
#                             import wandb
#                             if wandb.run is not None:
#                                 wandb.log({
#                                     "train/loss": loss.item(),
#                                     "train/lr": lr_scheduler.get_last_lr()[0],
#                                     "train/epoch": epoch,
#                                     "train/step": global_step,
#                                 }, step=global_step)
#                         except ImportError:
#                             pass

#                 # --- Save checkpoint ---
#                 save_every = config.get("save_every_n_steps", 500)
#                 if save_every and global_step % save_every == 0:
#                     if accelerator.is_main_process:
#                         save_lora(accelerator, transformer, config, global_step)

#                 # --- Validation ---
#                 val_every = config.get("validation_every_n_steps", 500)
#                 if val_every and global_step % val_every == 0 and val_control_images:
#                     if accelerator.is_main_process:
#                         val_pipeline = QwenImageEditPlusPipeline.from_pretrained(
#                             config.pretrained_model_name_or_path,
#                             transformer=accelerator.unwrap_model(transformer),
#                             torch_dtype=weight_dtype,
#                         )
#                         log_validation(
#                             val_pipeline, accelerator, config, epoch,
#                             global_step, val_control_images, val_captions
#                         )
#                         del val_pipeline
#                         torch.cuda.empty_cache()

#             # Check max steps
#             if config.get("max_train_steps") and global_step >= config.max_train_steps:
#                 break

#         # End of epoch
#         avg_epoch_loss = epoch_loss / max(global_step, 1)
#         logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")

#         if config.get("max_train_steps") and global_step >= config.max_train_steps:
#             break

#     # --- Final Save ---
#     if accelerator.is_main_process:
#         save_lora(accelerator, transformer, config, global_step, final=True)

#     logger.info("=" * 60)
#     logger.info("  TRAINING COMPLETE")
#     logger.info("=" * 60)
#     accelerator.end_training()


# def save_lora(accelerator, transformer, config, step, final=False):
#     """Save LoRA weights."""
#     suffix = "final" if final else f"step_{step}"
#     save_dir = Path(config.output_dir) / suffix

#     logger.info(f"Saving LoRA to {save_dir}...")

#     unwrapped = accelerator.unwrap_model(transformer)
#     unwrapped.save_pretrained(save_dir)

#     # Also save the config for reference
#     OmegaConf.save(config, save_dir / "train_config.yaml")

#     logger.info(f"LoRA saved to {save_dir}")


# # ============================================================================
# # ENTRY POINT
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser(description="Fine-tune Qwen-Image-Edit-2511 with LoRA")
#     parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

#     # Allow CLI overrides for common params
#     parser.add_argument("--output_dir", type=str, default=None)
#     parser.add_argument("--learning_rate", type=float, default=None)
#     parser.add_argument("--num_train_epochs", type=int, default=None)
#     parser.add_argument("--lora_rank", type=int, default=None)
#     parser.add_argument("--resolution", type=int, default=None)
#     parser.add_argument("--train_batch_size", type=int, default=None)

#     args = parser.parse_args()

#     # Load config
#     config = OmegaConf.load(args.config)

#     # Apply CLI overrides
#     for key in ["output_dir", "learning_rate", "num_train_epochs", "lora_rank",
#                 "resolution", "train_batch_size"]:
#         val = getattr(args, key)
#         if val is not None:
#             config[key] = val

#     # Create output dir
#     os.makedirs(config.output_dir, exist_ok=True)

#     # Save resolved config
#     OmegaConf.save(config, os.path.join(config.output_dir, "train_config.yaml"))

#     train(config)


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 LoRA Fine-Tuning Script
=============================================
Fine-tunes the Qwen-Image-Edit-2511 model using LoRA to learn
adding teacher comments (red ink annotations) to student answer sheets.

Dataset structure (from generate_dataset.py):
    qwen_edit_dataset/
    ├── targets/       (teacher images — annotated output)
    ├── control_1/     (student images — clean input)
    └── captions/      (editing instructions from comments)

Multi-GPU launch (recommended on 4×A5000):
    accelerate launch --num_processes 4 train_qwen_edit_lora.py

Single-GPU launch:
    python train_qwen_edit_lora.py
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from diffusers import QwenImageEditPlusPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

import platform
from pathlib import Path

# Windows torch wheels are built without libuv. torchrun/accelerate launch's
# TCPStore defaults to libuv and crashes with
# "use_libuv was requested but PyTorch was build without libuv support"
# unless this is set BEFORE torch.distributed gets imported. Harmless on Linux.
os.environ.setdefault("USE_LIBUV", "0")

# Raise PIL's decompression-bomb cap to accommodate large scanned answer sheets
# (~94 MP observed in this dataset). We deliberately set an explicit ceiling
# rather than disabling the check (None), so PIL still rejects pathologically
# large files. Adjust upward only if your real data exceeds this.
Image.MAX_IMAGE_PIXELS = 200_000_000  # ~14k x 14k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pinned revision for the community 4-bit quantized fork.
# Pinning protects against silent weight changes between training runs.
# If you switch --model_name, also update or remove this revision.
DEFAULT_MODEL_REVISION = "bbf3075e1aee61f91027fa212c9a3a03cd5dc7c4"


# ============================================================================
# DATASET
# ============================================================================

class QwenEditDataset(Dataset):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, control_dir, target_dir, caption_dir, resolution=1024):
        self.control_dir = Path(control_dir)
        self.target_dir = Path(target_dir)
        self.caption_dir = Path(caption_dir)
        self.resolution = resolution

        control_files = {
            f.stem: f for f in self.control_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        target_files = {
            f.stem: f for f in self.target_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        caption_files = {
            f.stem: f for f in self.caption_dir.iterdir()
            if f.suffix.lower() == ".txt"
        }

        self.keys = sorted(set(control_files) & set(target_files) & set(caption_files))
        self.control_files = control_files
        self.target_files = target_files
        self.caption_files = caption_files
        logger.info(f"Found {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def _load_pil(self, p):
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
    """
    Encode pixels with the QwenImage VAE.

    The QwenImage VAE is a video-style VAE (WAN 2.1) that expects
    5D input: [B, C, T, H, W]. For still images we add a temporal
    dimension of size 1, then squeeze it back out.

    Normalization uses per-channel mean/std from vae.config —
    NOT vae.config.scaling_factor (that's the SD/SDXL convention
    and does not apply to QwenImage).
    """
    pixels = pixels.to(device=vae.device, dtype=dtype)

    # [B, C, H, W] -> [B, C, 1, H, W]
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(2)

    latents = vae.encode(pixels).latent_dist.sample()

    # Per-channel normalization: (x - mean) / std
    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents.device, dtype=latents.dtype
    ).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std, device=latents.device, dtype=latents.dtype
    ).view(1, z_dim, 1, 1, 1)
    latents = (latents - latents_mean) / latents_std

    # Squeeze temporal dim back so downstream code sees [B, C, H, W].
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.squeeze(2)

    return latents


def pack_latents(latents):
    # [B, C, H, W] -> [B, (H/2)*(W/2), C*4] — Qwen transformer consumes a sequence of 2x2 patches.
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def compute_loss(pred, noise, latents):
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
    # On Windows there is no NCCL build, so we have to force gloo. Linux still
    # gets NCCL. Accelerate would otherwise default to NCCL whenever CUDA is
    # available and crash at init_process_group time on Windows multi-GPU.
    init_pg_backend = "gloo" if platform.system() == "Windows" else "nccl"
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[InitProcessGroupKwargs(backend=init_pg_backend)],
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

    # Optional memory saver: trade compute for VRAM. Important on 24 GB A5000s
    # at resolution 1024 with multi-image (control + noisy) sequence concat.
    if args.gradient_checkpointing:
        try:
            transformer.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on transformer.")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    # Do NOT call .to(accelerator.device) here. accelerator.prepare() (below)
    # wraps the transformer in DDP and moves it to the right device. Under gloo,
    # the DDP shape-verification allgather requires the params to be reachable
    # via the chosen backend; calling .to() before prepare() causes the WSL2-era
    # NCCL hang to recur as a gloo hang on native Windows.
    transformer.print_trainable_parameters()

    dataset = QwenEditDataset(
        args.control_dir, args.target_dir, args.caption_dir, args.resolution
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    num_update_steps_per_epoch = max(1, len(dataloader) // args.gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * args.epochs

    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps * accelerator.num_processes,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    logger.info("Starting training...")
    global_step = 0

    for epoch in range(args.epochs):
        transformer.train()
        progress = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )

        for batch in progress:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    target_latents = encode_vae(pipe.vae, batch["target"], weight_dtype)
                    control_latents = encode_vae(pipe.vae, batch["control"], weight_dtype)

                    # QwenImageEditPlusPipeline.encode_prompt expects:
                    #   prompt: List[str]                  (one per sample)
                    #   image:  List[List[PIL.Image]]      (outer = batch,
                    #                                       inner = ref images
                    #                                       for that sample)
                    # For single-control fine-tuning we wrap each PIL in its own list.
                    image_input = [[img] for img in batch["control_pil"]]

                    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                        prompt=batch["caption"],
                        image=image_input,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                    )

                bsz = target_latents.shape[0]
                t = torch.rand(bsz, device=target_latents.device, dtype=weight_dtype)
                sigmas = t.view(-1, 1, 1, 1)
                noise = torch.randn_like(target_latents)
                noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise

                # Pack latents and concat control along the sequence dim (not channel dim).
                noisy_packed = pack_latents(noisy_latents)
                control_packed = pack_latents(control_latents)
                hidden_states = torch.cat([noisy_packed, control_packed], dim=1)

                pred = transformer(
                    hidden_states=hidden_states,
                    timestep=t * 1000.0,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    return_dict=False,
                )[0]

                target_seq_len = noisy_packed.shape[1]
                pred_target = pred[:, :target_seq_len]
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
                    progress.set_postfix(
                        loss=f"{logs['train/loss']:.4f}",
                        lr=f"{logs['train/lr']:.2e}",
                    )
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
    # Required on Windows; harmless on Linux. Without this, child processes
    # spawned by DataLoader workers or accelerate launch can re-enter __main__
    # before the parent has finished bootstrapping.
    import multiprocessing
    multiprocessing.freeze_support()

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _DATA = _REPO_ROOT / "qwen_edit_dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="toandev/Qwen-Image-Edit-2511-4bit",
        help="4-bit quantized base. Required on 24 GB A5000s — full-precision "
             "Qwen-Image-Edit-2511 is ~41 GB and does not fit.",
    )
    parser.add_argument(
        "--model_revision",
        default=DEFAULT_MODEL_REVISION,
        help="HF Hub revision (commit hash) to pin. "
             "Pass '' to disable pinning. Update if you switch --model_name.",
    )
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
    parser.add_argument(
        "--target_modules", nargs="+", default=["to_k", "to_q", "to_v", "to_out.0"]
    )
    parser.add_argument("--resolution", type=int, default=768,
                        help="768 fits 24 GB A5000 with the 4-bit base + LoRA + "
                             "gradient checkpointing. 1024 lives close to the OOM "
                             "ceiling; only raise it after a successful 768 run.")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for VRAM. Recommended on 24 GB cards at res 1024.")
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 is the safe Windows default — DataLoader workers "
                             "spawn via 'spawn' on Windows and re-import the "
                             "module per worker; >0 only after a clean baseline run.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)