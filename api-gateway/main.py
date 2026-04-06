"""
API Gateway Service
-------------------
The central entry point for all client requests.
Routes traffic to Model Management and GPU Inference services.
Handles authentication, rate limiting, and request logging.
"""

import time
import uuid
import httpx
import redis
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("api-gateway")

# ── Environment ───────────────────────────────────────────────────────────────
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8001")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference-service:8002")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# ── Redis client ──────────────────────────────────────────────────────────────
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML-as-a-Service API Gateway",
    description="Distributed GPU-Accelerated AI Inference Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    version: Optional[str] = "latest"
    priority: Optional[int] = 1          # 1=low, 5=high


class InferenceResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    queued_at: Optional[float] = None
    message: Optional[str] = None


# ── Middleware: request timing & tracing ──────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({duration*1000:.1f}ms) [req={request_id}]"
    )
    return response


# ── Rate limiter (simple sliding-window via Redis) ────────────────────────────
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

async def rate_limit(request: Request):
    client_ip = request.client.host
    key = f"rl:{client_ip}"
    try:
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        results = pipe.execute()
        count = results[0]
        if count > RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests/minute.",
            )
    except redis.RedisError:
        logger.warning("Redis unavailable – skipping rate limit check")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    services = {}
    async with httpx.AsyncClient(timeout=3) as client:
        for name, url in [
            ("model-service", MODEL_SERVICE_URL),
            ("inference-service", INFERENCE_SERVICE_URL),
        ]:
            try:
                r = await client.get(f"{url}/health")
                services[name] = "ok" if r.status_code == 200 else "degraded"
            except Exception:
                services[name] = "unreachable"

    try:
        redis_client.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "unreachable"

    overall = "healthy" if all(v == "ok" for v in services.values()) else "degraded"
    return {"status": overall, "services": services, "timestamp": time.time()}


# ── Model Management Proxy ────────────────────────────────────────────────────
@app.post("/models/upload", tags=["Models"])
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = "my_model",
    version: str = "1.0.0",
    description: str = "",
    _: None = Depends(rate_limit),
):
    """Upload a PyTorch .pt/.pth model file."""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            content = await file.read()
            files = {"file": (file.filename, content, file.content_type)}
            params = {"model_name": model_name, "version": version, "description": description}
            r = await client.post(f"{MODEL_SERVICE_URL}/models/upload", files=files, params=params)
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Model service unavailable: {e}")


@app.get("/models", tags=["Models"])
async def list_models(_: None = Depends(rate_limit)):
    """List all registered models."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{MODEL_SERVICE_URL}/models")
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.get("/models/{model_id}", tags=["Models"])
async def get_model(model_id: str, _: None = Depends(rate_limit)):
    """Get metadata for a specific model."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{MODEL_SERVICE_URL}/models/{model_id}")
            if r.status_code == 404:
                raise HTTPException(status_code=404, detail="Model not found")
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str, _: None = Depends(rate_limit)):
    """Delete a model."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.delete(f"{MODEL_SERVICE_URL}/models/{model_id}")
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=str(e))


# ── Inference ─────────────────────────────────────────────────────────────────
@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer(payload: InferenceRequest, _: None = Depends(rate_limit)):
    """
    Submit an inference request.
    The request is enqueued in Redis and processed by the GPU inference worker.
    """
    request_id = str(uuid.uuid4())
    task = {
        "request_id": request_id,
        "model_id": payload.model_id,
        "version": payload.version,
        "input_data": payload.input_data,
        "priority": payload.priority,
        "submitted_at": time.time(),
    }

    # Push to Redis queue (priority-aware – higher priority = separate queue)
    queue_key = f"inference:queue:p{payload.priority}"
    try:
        import json
        redis_client.lpush(queue_key, json.dumps(task))
        redis_client.setex(f"task:{request_id}:status", 3600, "queued")
        logger.info(f"Enqueued task {request_id} for model {payload.model_id}")
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")

    return InferenceResponse(
        request_id=request_id,
        status="queued",
        queued_at=task["submitted_at"],
        message="Request enqueued. Poll /infer/status/{request_id} for result.",
    )


@app.get("/infer/status/{request_id}", tags=["Inference"])
async def get_inference_status(request_id: str):
    """Poll inference result by request ID."""
    import json
    try:
        status = redis_client.get(f"task:{request_id}:status")
        if not status:
            raise HTTPException(status_code=404, detail="Request ID not found or expired")

        result_raw = redis_client.get(f"task:{request_id}:result")
        result = json.loads(result_raw) if result_raw else None

        return {
            "request_id": request_id,
            "status": status,
            "result": result,
        }
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/infer/sync", tags=["Inference"])
async def infer_sync(payload: InferenceRequest, _: None = Depends(rate_limit)):
    """Synchronous inference (bypasses queue – for low-latency use cases)."""
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{INFERENCE_SERVICE_URL}/infer",
                json=payload.dict(),
            )
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Inference service unavailable: {e}")


# ── Metrics summary ───────────────────────────────────────────────────────────
@app.get("/metrics/summary", tags=["Metrics"])
async def metrics_summary():
    """Return basic queue depth and task stats."""
    try:
        queues = {}
        for p in range(1, 6):
            key = f"inference:queue:p{p}"
            queues[f"priority_{p}"] = redis_client.llen(key)
        return {"queue_depths": queues, "timestamp": time.time()}
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=str(e))
