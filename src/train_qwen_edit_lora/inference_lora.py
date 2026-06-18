# #!/usr/bin/env python3
# """
# Inference with Fine-Tuned Qwen-Image-Edit-2511 LoRA
# ====================================================
# Load the base model + trained LoRA weights and run image editing.

# Usage:
#     python inference_lora.py --lora_path ./output/qwen_edit_lora/final
#     python inference_lora.py --lora_path ./output/qwen_edit_lora/step_1000
# """

# import argparse
# import os
# import torch
# from PIL import Image
# from diffusers import QwenImageEditPlusPipeline


# def load_pipeline_with_lora(lora_path: str, device: str = "cuda"):
#     """Load the base pipeline and apply trained LoRA weights."""

#     print("Loading base pipeline...")
#     pipeline = QwenImageEditPlusPipeline.from_pretrained(
#         "Qwen/Qwen-Image-Edit-2511",
#         torch_dtype=torch.bfloat16,
#     )

#     print(f"Loading LoRA weights from {lora_path}...")
#     pipeline.load_lora_weights(lora_path)

#     pipeline.to(device)
#     pipeline.set_progress_bar_config(disable=None)
#     print("Pipeline ready!")

#     return pipeline


# def edit_image(
#     pipeline,
#     image_path: str,
#     prompt: str,
#     output_path: str = "output_edited.png",
#     seed: int = 0,
#     num_inference_steps: int = 40,
#     true_cfg_scale: float = 4.0,
# ):
#     """Run image editing with the LoRA-enhanced pipeline."""

#     image = Image.open(image_path).convert("RGB")
#     print(f"Input: {image_path} ({image.size})")
#     print(f"Prompt: {prompt}")

#     inputs = {
#         "image": [image],
#         "prompt": prompt,
#         "generator": torch.manual_seed(seed),
#         "true_cfg_scale": true_cfg_scale,
#         "negative_prompt": " ",
#         "num_inference_steps": num_inference_steps,
#         "guidance_scale": 1.0,
#         "num_images_per_prompt": 1,
#     }

#     with torch.inference_mode():
#         output = pipeline(**inputs)
#         result = output.images[0]

#     result.save(output_path)
#     print(f"Saved: {output_path}")
#     return result


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lora_path", required=True, help="Path to LoRA checkpoint")
#     parser.add_argument("--image", required=True, help="Input student image path")
#     parser.add_argument("--prompt", required=True, help="Edit instruction / comments")
#     parser.add_argument("--output", default="output_edited.png", help="Output path")
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--steps", type=int, default=40)
#     parser.add_argument("--cfg", type=float, default=4.0)

#     args = parser.parse_args()

#     pipeline = load_pipeline_with_lora(args.lora_path)

#     edit_image(
#         pipeline,
#         image_path=args.image,
#         prompt=args.prompt,
#         output_path=args.output,
#         seed=args.seed,
#         num_inference_steps=args.steps,
#         true_cfg_scale=args.cfg,
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
import os
import torch
from PIL import Image
from peft import PeftModel
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


def _check_adapter_dir(lora_path):
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(
            f"LoRA directory not found: {lora_path}\n"
            "Pass --lora_path the folder that contains adapter_config.json and "
            "adapter_model.safetensors (e.g. .../lora_output/final)."
        )
    if not os.path.isfile(os.path.join(lora_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"{lora_path} has no adapter_config.json -- it is not a PEFT adapter dir."
        )


def load_pipeline(lora_path, device="cuda", quant="4bit"):
    _check_adapter_dir(lora_path)

    if quant == "4bit":
        # The bf16 transformer (~40GB) does not fit in 24GB VRAM; on WSL it spills
        # into system RAM and crawls. Load it in 4-bit (NF4) so it fits in VRAM, and
        # offload the text encoder / VAE to CPU between calls.
        from diffusers import QwenImageTransformer2DModel
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

        bnb = DiffusersBnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        # Attach the trained LoRA on top of the 4-bit transformer (QLoRA-style).
        # Do NOT merge_and_unload(): you cannot merge a LoRA into 4-bit weights.
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        # Offload manages device placement -- do NOT call pipe.to(device) after this.
        pipe.enable_model_cpu_offload()
    else:
        # Full-precision path -- needs ~48GB+ VRAM. Merge the LoRA for fastest inference.
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        pipe.transformer = pipe.transformer.merge_and_unload()
        pipe.to(device)

    pipe.set_progress_bar_config(disable=None)
    return pipe

def run_inference(pipeline, image_path, prompt, out_file, seed, steps, true_cfg):
    img = Image.open(image_path).convert("RGB")
    gen = torch.manual_seed(seed)

    with torch.inference_mode():
        output = pipeline(
            image=[img],
            prompt=prompt,
            generator=gen,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg,
            negative_prompt=" ",
        )

    output.images[0].save(out_file)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="edited.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--quant", choices=["4bit", "none"], default="4bit",
                        help="4bit fits ~24GB VRAM (default); 'none' needs ~48GB+.")
    args = parser.parse_args()

    # Allow --prompt to be either literal text or a path to a .txt caption file.
    prompt = args.prompt
    if os.path.isfile(prompt):
        with open(prompt, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

    pipeline = load_pipeline(args.lora_path, quant=args.quant)
    run_inference(pipeline, args.input_image, prompt, args.output,
                  args.seed, args.steps, args.cfg)