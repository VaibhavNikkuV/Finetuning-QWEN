# #!/usr/bin/env python3
# """
# Qwen Image Edit Server with LoRA Support
# =========================================
# Extended version of qwen_server.py that loads fine-tuned LoRA weights.

# Usage:
#     python qwen_server_lora.py --port 6000 --lora_path ./output/qwen_edit_lora/final
# """

# import argparse
# import base64
# import io
# import logging
# import time
# from typing import Optional, List
# from contextlib import asynccontextmanager

# import torch
# from PIL import Image
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# # --- Request/Response Models (same as original) ---

# class ImageEditRequest(BaseModel):
#     image_base64: str
#     prompt: str
#     negative_prompt: Optional[str] = " "
#     num_inference_steps: Optional[int] = 40
#     guidance_scale: Optional[float] = 1.0
#     true_cfg_scale: Optional[float] = 4.0
#     seed: Optional[int] = 0


# class ImageEditResponse(BaseModel):
#     success: bool
#     image_base64: Optional[str] = None
#     error: Optional[str] = None
#     processing_time: Optional[float] = None


# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool
#     lora_loaded: bool
#     lora_path: Optional[str] = None
#     gpu_count: int
#     gpu_info: List[dict]


# # --- Model Holder with LoRA ---

# class ModelHolder:
#     def __init__(self):
#         self.pipeline = None
#         self.loaded = False
#         self.lora_loaded = False
#         self.lora_path = None

#     def load(self, lora_path: Optional[str] = None, device_map: str = "balanced"):
#         if self.loaded:
#             return

#         logger.info("=" * 60)
#         logger.info("  LOADING QWEN IMAGE EDIT MODEL + LoRA")
#         logger.info("=" * 60)

#         gpu_count = torch.cuda.device_count()
#         logger.info(f"Available GPUs: {gpu_count}")
#         for i in range(gpu_count):
#             props = torch.cuda.get_device_properties(i)
#             logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

#         from diffusers import QwenImageEditPlusPipeline

#         start_time = time.time()
#         self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
#             "Qwen/Qwen-Image-Edit-2511",
#             torch_dtype=torch.bfloat16,
#             device_map=device_map,
#             low_cpu_mem_usage=True,
#         )
#         logger.info(f"Base pipeline loaded in {time.time() - start_time:.2f}s")

#         # Load LoRA if specified
#         if lora_path:
#             logger.info(f"Loading LoRA from {lora_path}...")
#             self.pipeline.load_lora_weights(lora_path)
#             self.lora_loaded = True
#             self.lora_path = lora_path
#             logger.info("LoRA weights loaded!")

#         self.pipeline.set_progress_bar_config(disable=None)
#         self.loaded = True
#         logger.info("=" * 60)
#         logger.info("  MODEL READY" + (" (with LoRA)" if lora_path else ""))
#         logger.info("=" * 60)

#     def generate(self, image, prompt, negative_prompt=" ",
#                  num_inference_steps=40, guidance_scale=1.0,
#                  true_cfg_scale=4.0, seed=0):
#         if not self.loaded:
#             raise RuntimeError("Model not loaded")

#         inputs = {
#             "image": [image],
#             "prompt": prompt,
#             "generator": torch.manual_seed(seed),
#             "true_cfg_scale": true_cfg_scale,
#             "negative_prompt": negative_prompt,
#             "num_inference_steps": num_inference_steps,
#             "guidance_scale": guidance_scale,
#             "num_images_per_prompt": 1,
#         }

#         with torch.inference_mode():
#             output = self.pipeline(**inputs)
#             return output.images[0]


# model_holder = ModelHolder()
# _lora_path_arg = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     model_holder.load(lora_path=_lora_path_arg)
#     yield
#     logger.info("Shutting down...")


# app = FastAPI(
#     title="Qwen Image Edit Server (LoRA)",
#     description="API server for fine-tuned Qwen-Image-Edit-2511",
#     version="1.1.0",
#     lifespan=lifespan,
# )


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     gpu_info = []
#     for i in range(torch.cuda.device_count()):
#         props = torch.cuda.get_device_properties(i)
#         gpu_info.append({
#             "index": i,
#             "name": props.name,
#             "total_memory_gb": round(props.total_memory / 1024**3, 2),
#             "allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
#         })
#     return HealthResponse(
#         status="healthy" if model_holder.loaded else "loading",
#         model_loaded=model_holder.loaded,
#         lora_loaded=model_holder.lora_loaded,
#         lora_path=model_holder.lora_path,
#         gpu_count=torch.cuda.device_count(),
#         gpu_info=gpu_info,
#     )


# @app.post("/generate", response_model=ImageEditResponse)
# async def generate_image(request: ImageEditRequest):
#     if not model_holder.loaded:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     start_time = time.time()
#     try:
#         image_data = base64.b64decode(request.image_base64)
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")

#         output_image = model_holder.generate(
#             image=image,
#             prompt=request.prompt,
#             negative_prompt=request.negative_prompt or " ",
#             num_inference_steps=request.num_inference_steps or 40,
#             guidance_scale=request.guidance_scale or 1.0,
#             true_cfg_scale=request.true_cfg_scale or 4.0,
#             seed=request.seed or 0,
#         )

#         buffer = io.BytesIO()
#         output_image.save(buffer, format="PNG")
#         output_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

#         return ImageEditResponse(
#             success=True,
#             image_base64=output_base64,
#             processing_time=time.time() - start_time,
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"Error: {e}")
#         return ImageEditResponse(
#             success=False, error=str(e),
#             processing_time=time.time() - start_time,
#         )


# def main():
#     global _lora_path_arg

#     parser = argparse.ArgumentParser(description="Qwen Image Edit Server with LoRA")
#     parser.add_argument("--host", default="0.0.0.0")
#     parser.add_argument("--port", type=int, default=6000)
#     parser.add_argument("--lora_path", type=str, default=None,
#                         help="Path to fine-tuned LoRA weights")
#     parser.add_argument("--workers", type=int, default=1)
#     args = parser.parse_args()

#     _lora_path_arg = args.lora_path

#     logger.info(f"Server: {args.host}:{args.port}")
#     if args.lora_path:
#         logger.info(f"LoRA: {args.lora_path}")

#     uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import argparse
import base64
import io
import logging
import os

import torch
import uvicorn
from diffusers import QwenImageEditPlusPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

class EditRequest(BaseModel):
    image_base64: str
    prompt: str
    negative_prompt: str = " "
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    true_cfg_scale: float = 4.0
    seed: int | None = None

class EditResponse(BaseModel):
    success: bool
    image_base64: str | None = None
    error: str | None = None

app = FastAPI(title="Qwen Image Edit + LoRA Server")

model_state = {"pipe": None, "lora_path": None}

@app.on_event("startup")
async def load_base():
    logger.info("Loading Qwen base model...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe.set_progress_bar_config(disable=None)

    lora_path = os.environ.get("QWEN_LORA_PATH")
    if lora_path:
        logger.info(f"Loading LoRA from {lora_path}...")
        pipe.load_lora_weights(lora_path)
        model_state["lora_path"] = lora_path
        logger.info("LoRA weights loaded")

    model_state["pipe"] = pipe
    logger.info("Model ready" + (" (with LoRA)" if lora_path else ""))

@app.post("/generate", response_model=EditResponse)
async def generate(req: EditRequest):
    if model_state["pipe"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        gen = torch.manual_seed(req.seed) if req.seed is not None else None

        pipe = model_state["pipe"]
        with torch.inference_mode():
            out = pipe(
                image=[img],
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                true_cfg_scale=req.true_cfg_scale,
                guidance_scale=req.guidance_scale,
                generator=gen,
            )

        buf = io.BytesIO()
        out.images[0].save(buf, format="PNG")
        enc = base64.b64encode(buf.getvalue()).decode("utf-8")

        return EditResponse(success=True, image_base64=enc)

    except Exception as e:
        logger.error("Error", exc_info=True)
        return EditResponse(success=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen Image Edit + LoRA server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--lora_path", default=None, help="Path to trained LoRA checkpoint")
    args = parser.parse_args()

    if args.lora_path:
        os.environ["QWEN_LORA_PATH"] = args.lora_path
        logger.info(f"LoRA path set: {args.lora_path}")

    uvicorn.run(app, host=args.host, port=args.port)