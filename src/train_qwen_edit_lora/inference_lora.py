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
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def load_pipeline(lora_path, device="cuda"):
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(lora_path)
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
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=4.0)
    args = parser.parse_args()

    pipeline = load_pipeline(args.lora_path)
    run_inference(pipeline, args.input_image, args.prompt, args.output,
                  args.seed, args.steps, args.cfg)