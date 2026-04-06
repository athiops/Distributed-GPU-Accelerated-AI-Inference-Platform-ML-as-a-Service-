"""
Model Management Service
-------------------------
Handles model upload, storage, versioning, and metadata.
Models are stored on a shared volume (or S3 in production).
"""

import os
import uuid
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("model-service")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_STORE = Path(os.getenv("MODEL_STORE", "/app/model_store"))
MODEL_STORE.mkdir(parents=True, exist_ok=True)
METADATA_FILE = MODEL_STORE / "registry.json"

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Model Management Service",
    description="Upload, version, and manage ML models",
    version="1.0.0",
)


# ── Registry helpers ──────────────────────────────────────────────────────────
def load_registry() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    with open(METADATA_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Schemas ───────────────────────────────────────────────────────────────────
class ModelMeta(BaseModel):
    model_id: str
    name: str
    version: str
    description: str
    file_path: str
    file_size_mb: float
    checksum: str
    framework: str
    uploaded_at: float
    status: str  # "ready" | "loading" | "error"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "model-service"}


@app.post("/models/upload", response_model=ModelMeta)
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = Query("my_model", description="Human-readable name"),
    version: str = Query("1.0.0", description="Semantic version"),
    description: str = Query("", description="Short description"),
    framework: str = Query("pytorch", description="pytorch | onnx | tensorflow"),
):
    """Upload a model file. Supports .pt, .pth, .onnx, .pb."""
    allowed_exts = {".pt", ".pth", ".onnx", ".pb", ".h5"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_exts}",
        )

    model_id = str(uuid.uuid4())[:8]
    dest_dir = MODEL_STORE / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    # Stream file to disk
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    checksum = sha256_file(dest_path)

    meta = {
        "model_id": model_id,
        "name": model_name,
        "version": version,
        "description": description,
        "file_path": str(dest_path),
        "file_size_mb": round(size_mb, 3),
        "checksum": checksum,
        "framework": framework,
        "uploaded_at": time.time(),
        "status": "ready",
    }

    registry = load_registry()
    registry[model_id] = meta
    save_registry(registry)

    logger.info(f"Registered model {model_id} ({model_name} v{version})")
    return ModelMeta(**meta)


@app.get("/models", response_model=List[ModelMeta])
async def list_models():
    """Return all registered models."""
    registry = load_registry()
    return [ModelMeta(**v) for v in registry.values()]


@app.get("/models/{model_id}", response_model=ModelMeta)
async def get_model(model_id: str):
    """Get metadata for a specific model."""
    registry = load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelMeta(**registry[model_id])


@app.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download the model file."""
    registry = load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")
    path = Path(registry[model_id]["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file missing from disk")
    return FileResponse(path, filename=path.name)


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Remove a model from registry and disk."""
    registry = load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")

    path = Path(registry[model_id]["file_path"]).parent
    shutil.rmtree(path, ignore_errors=True)
    del registry[model_id]
    save_registry(registry)
    logger.info(f"Deleted model {model_id}")
    return {"status": "deleted", "model_id": model_id}


@app.get("/models/{model_id}/versions")
async def list_versions(model_id: str):
    """List all versions for a model name."""
    registry = load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")

    name = registry[model_id]["name"]
    versions = [
        {"model_id": k, "version": v["version"], "uploaded_at": v["uploaded_at"]}
        for k, v in registry.items()
        if v["name"] == name
    ]
    return {"name": name, "versions": versions}
