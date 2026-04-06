"""
GPU Inference Service
----------------------
Loads PyTorch models and runs inference on NVIDIA GPUs.
Supports both synchronous REST calls and async queue workers.
Includes request batching for throughput optimization.
"""

import os
import json
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import httpx
import redis
import numpy as np
import io
import base64

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("inference-service")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_STORE = Path(os.getenv("MODEL_STORE", "/app/model_store"))
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8001")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "50"))

# ── Device selection (GPU → CPU fallback) ─────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ── Redis client ──────────────────────────────────────────────────────────────
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# ── In-memory model cache ─────────────────────────────────────────────────────
model_cache: Dict[str, nn.Module] = {}
model_cache_lock = threading.Lock()

# ── ImageNet class labels (top-5 visible) ────────────────────────────────────
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
IMAGENET_LABELS: List[str] = []

def load_imagenet_labels():
    global IMAGENET_LABELS
    try:
        import urllib.request
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=5) as r:
            IMAGENET_LABELS = json.loads(r.read().decode())
        logger.info("Loaded ImageNet labels")
    except Exception:
        IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
        logger.warning("Could not fetch ImageNet labels — using placeholders")


# ── Image preprocessing pipeline ─────────────────────────────────────────────
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(b64_image: str) -> torch.Tensor:
    """Decode base64 image → normalised tensor."""
    raw = base64.b64decode(b64_image)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return IMAGENET_TRANSFORM(img)


def get_or_load_model(model_id: str) -> nn.Module:
    """Return cached model or load from disk."""
    with model_cache_lock:
        if model_id in model_cache:
            return model_cache[model_id]

        # Fetch metadata from model service
        try:
            resp = httpx.get(f"{MODEL_SERVICE_URL}/models/{model_id}", timeout=10)
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not registered")
            meta = resp.json()
        except httpx.RequestError as e:
            # Fall back to demo ResNet50 for development
            logger.warning(f"Model service unreachable ({e}) – loading demo ResNet50")
            meta = None

        if meta:
            model_path = Path(meta["file_path"])
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Model file missing from disk")

            logger.info(f"Loading model {model_id} from {model_path}")
            model = torch.load(model_path, map_location=DEVICE)
        else:
            # Demo mode: pretrained ResNet50
            logger.info("Loading demo ResNet50 (ImageNet pretrained)")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        model.eval()
        model.to(DEVICE)

        # Optional: TorchScript compile for speedup
        try:
            dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            model = torch.jit.trace(model, dummy)
            logger.info(f"TorchScript compiled model {model_id}")
        except Exception as e:
            logger.warning(f"TorchScript compile failed ({e}) – using eager mode")

        model_cache[model_id] = model
        return model


@torch.no_grad()
def run_inference_batch(model: nn.Module, tensors: List[torch.Tensor]) -> List[Dict]:
    """Run a batch of tensors through the model."""
    batch = torch.stack(tensors).to(DEVICE)  # (N, 3, 224, 224)

    start = time.perf_counter()
    logits = model(batch)                     # (N, 1000)
    latency_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1)
    top5_probs, top5_idx = torch.topk(probs, 5, dim=1)

    results = []
    for i in range(len(tensors)):
        predictions = [
            {
                "rank": r + 1,
                "class_id": int(top5_idx[i, r]),
                "label": IMAGENET_LABELS[int(top5_idx[i, r])] if IMAGENET_LABELS else f"class_{int(top5_idx[i, r])}",
                "confidence": float(top5_probs[i, r]),
            }
            for r in range(5)
        ]
        results.append({
            "predictions": predictions,
            "latency_ms": round(latency_ms / len(tensors), 2),
            "device": str(DEVICE),
        })

    return results


# ── Queue worker (background thread) ─────────────────────────────────────────
def queue_worker():
    """Continuously drain inference queues from Redis (highest priority first)."""
    logger.info("Queue worker started")
    while True:
        try:
            # Priority 5 → 1
            for p in range(5, 0, -1):
                queue_key = f"inference:queue:p{p}"
                raw = redis_client.rpop(queue_key)
                if not raw:
                    continue

                task = json.loads(raw)
                request_id = task["request_id"]
                model_id = task["model_id"]
                input_data = task["input_data"]

                logger.info(f"Processing task {request_id} (model={model_id})")
                redis_client.set(f"task:{request_id}:status", "processing")

                try:
                    model = get_or_load_model(model_id)

                    if "image_b64" in input_data:
                        tensor = preprocess_image(input_data["image_b64"])
                        results = run_inference_batch(model, [tensor])
                        result = results[0]
                    else:
                        # Generic tensor input
                        data = torch.tensor(input_data.get("tensor", [[0.0] * 224])).float()
                        result = {"raw_output": data.tolist(), "device": str(DEVICE)}

                    redis_client.setex(f"task:{request_id}:result", 3600, json.dumps(result))
                    redis_client.setex(f"task:{request_id}:status", 3600, "completed")
                    logger.info(f"Task {request_id} completed")

                except Exception as e:
                    logger.error(f"Task {request_id} failed: {e}")
                    redis_client.setex(f"task:{request_id}:status", 3600, "failed")
                    redis_client.setex(
                        f"task:{request_id}:result", 3600,
                        json.dumps({"error": str(e)}),
                    )
                break  # processed one task; loop again for next

        except redis.RedisError as e:
            logger.error(f"Redis error in worker: {e}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(1)

        time.sleep(0.01)  # 10 ms poll


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_imagenet_labels()
    # Pre-warm demo model
    try:
        get_or_load_model("demo")
    except Exception:
        pass
    # Start background worker thread
    t = threading.Thread(target=queue_worker, daemon=True)
    t.start()
    yield
    logger.info("Shutting down inference service")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GPU Inference Service",
    description="CUDA-accelerated PyTorch inference with batching",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class InferencePayload(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    version: Optional[str] = "latest"


class BatchPayload(BaseModel):
    model_id: str
    inputs: List[Dict[str, Any]]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
    return {
        "status": "ok",
        "service": "inference-service",
        "device": str(DEVICE),
        "loaded_models": list(model_cache.keys()),
        "gpu": gpu_info,
    }


@app.post("/infer")
async def infer(payload: InferencePayload):
    """Single synchronous inference request."""
    model = get_or_load_model(payload.model_id)
    input_data = payload.input_data

    if "image_b64" in input_data:
        try:
            tensor = preprocess_image(input_data["image_b64"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

        results = run_inference_batch(model, [tensor])
        return {
            "model_id": payload.model_id,
            "request_id": str(uuid.uuid4()),
            **results[0],
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide 'image_b64' (base64-encoded JPEG/PNG) in input_data",
        )


@app.post("/infer/batch")
async def infer_batch(payload: BatchPayload):
    """Batched inference – process multiple images in one forward pass."""
    if not payload.inputs:
        raise HTTPException(status_code=400, detail="inputs list is empty")
    if len(payload.inputs) > BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit ({BATCH_SIZE})",
        )

    model = get_or_load_model(payload.model_id)
    tensors = []
    for i, inp in enumerate(payload.inputs):
        if "image_b64" not in inp:
            raise HTTPException(status_code=400, detail=f"Input {i} missing 'image_b64'")
        try:
            tensors.append(preprocess_image(inp["image_b64"]))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Input {i} decode error: {e}")

    results = run_inference_batch(model, tensors)
    return {
        "model_id": payload.model_id,
        "batch_size": len(tensors),
        "results": results,
    }


@app.post("/models/{model_id}/warm")
async def warm_model(model_id: str, background_tasks: BackgroundTasks):
    """Pre-load a model into GPU memory."""
    if model_id in model_cache:
        return {"status": "already_loaded", "model_id": model_id}
    background_tasks.add_task(get_or_load_model, model_id)
    return {"status": "warming", "model_id": model_id}


@app.delete("/models/{model_id}/cache")
async def evict_model(model_id: str):
    """Evict a model from GPU memory cache."""
    with model_cache_lock:
        if model_id in model_cache:
            del model_cache[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"status": "evicted", "model_id": model_id}
    return {"status": "not_in_cache", "model_id": model_id}


@app.get("/gpu/stats")
async def gpu_stats():
    """Return live GPU utilization and memory stats."""
    if not torch.cuda.is_available():
        return {"gpu_available": False, "device": "cpu"}

    stats = {
        "gpu_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        stats["devices"].append({
            "id": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "allocated_memory_gb": round(torch.cuda.memory_allocated(i) / 1e9, 3),
            "reserved_memory_gb": round(torch.cuda.memory_reserved(i) / 1e9, 3),
            "multi_processor_count": props.multi_processor_count,
        })
    return stats
