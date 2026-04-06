"""
Sample Inference Script
-----------------------
Demonstrates calling the platform's API for inference.

Usage:
    python sample_inference.py --image path/to/image.jpg --gateway http://localhost:8000
    python sample_inference.py --demo               (uses a generated synthetic image)
"""

import argparse
import base64
import json
import time
import sys
import io

import httpx
from PIL import Image, ImageDraw
import numpy as np


def create_demo_image() -> str:
    """Create a simple synthetic image and return as base64."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_sync_inference(gateway_url: str, model_id: str, b64_image: str):
    """Run synchronous inference via /infer/sync."""
    payload = {
        "model_id": model_id,
        "input_data": {"image_b64": b64_image},
    }
    print(f"\n[→] POST {gateway_url}/infer/sync")
    start = time.perf_counter()
    r = httpx.post(f"{gateway_url}/infer/sync", json=payload, timeout=60)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"[←] Status: {r.status_code} ({elapsed:.0f}ms round-trip)")

    if r.status_code != 200:
        print(f"[!] Error: {r.text}")
        return

    result = r.json()
    print(f"\n{'='*55}")
    print(f"  Sync Inference Result")
    print(f"{'='*55}")
    print(f"  Model    : {model_id}")
    print(f"  Device   : {result.get('device', 'unknown')}")
    print(f"  Latency  : {result.get('latency_ms', '?')}ms (server-side)")
    print(f"  RTT      : {elapsed:.0f}ms")
    print(f"\n  Top-5 Predictions:")
    for pred in result.get("predictions", []):
        bar = "█" * int(pred["confidence"] * 20)
        print(f"    {pred['rank']}. {pred['label']:<30} {pred['confidence']:.4f}  {bar}")
    print(f"{'='*55}\n")


def run_async_inference(gateway_url: str, model_id: str, b64_image: str):
    """Submit to async queue and poll for result."""
    payload = {
        "model_id": model_id,
        "input_data": {"image_b64": b64_image},
        "priority": 5,
    }
    print(f"\n[→] POST {gateway_url}/infer (async queue)")
    r = httpx.post(f"{gateway_url}/infer", json=payload, timeout=30)

    if r.status_code != 200:
        print(f"[!] Error: {r.text}")
        return

    resp = r.json()
    request_id = resp["request_id"]
    print(f"[✓] Enqueued  request_id={request_id}")

    # Poll until done
    print("[…] Polling for result ", end="", flush=True)
    for _ in range(60):
        time.sleep(1)
        status_r = httpx.get(f"{gateway_url}/infer/status/{request_id}", timeout=10)
        state = status_r.json()
        print(".", end="", flush=True)

        if state["status"] in ("completed", "failed"):
            break

    print()
    result = state.get("result", {})
    if state["status"] == "completed":
        print(f"\n{'='*55}")
        print(f"  Async Inference Result  (request_id={request_id})")
        print(f"{'='*55}")
        for pred in result.get("predictions", []):
            bar = "█" * int(pred["confidence"] * 20)
            print(f"    {pred['rank']}. {pred['label']:<30} {pred['confidence']:.4f}  {bar}")
        print(f"{'='*55}\n")
    else:
        print(f"[!] Task failed: {result}")


def run_batch_inference(gateway_url: str, model_id: str, b64_images: list):
    """Submit a batch of images for inference."""
    payload = {
        "model_id": model_id,
        "inputs": [{"image_b64": b64} for b64 in b64_images],
    }
    print(f"\n[→] POST {gateway_url}/infer/sync (batch of {len(b64_images)})")
    # For batch, call the inference service directly (or extend gateway to proxy batch)
    r = httpx.post(
        f"{gateway_url.replace(':8000', ':8002')}/infer/batch",
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        print(f"[!] Error: {r.text}")
        return
    data = r.json()
    print(f"[✓] Batch results ({data['batch_size']} images):")
    for i, res in enumerate(data["results"]):
        top1 = res["predictions"][0]
        print(f"  [{i}] {top1['label']:<30} conf={top1['confidence']:.4f}  lat={res['latency_ms']}ms")


def check_health(gateway_url: str):
    print(f"\n[→] GET {gateway_url}/health")
    r = httpx.get(f"{gateway_url}/health", timeout=10)
    health = r.json()
    status = health["status"]
    icon = "✅" if status == "healthy" else "⚠️"
    print(f"{icon}  Platform status: {status.upper()}")
    for svc, state in health["services"].items():
        icon = "✅" if state == "ok" else "❌"
        print(f"   {icon} {svc}: {state}")


def main():
    parser = argparse.ArgumentParser(description="ML Platform Inference Demo")
    parser.add_argument("--gateway", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--model-id", default="demo", help="Model ID (use 'demo' for built-in)")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo image")
    parser.add_argument("--async-mode", action="store_true", help="Use async queue mode")
    parser.add_argument("--batch", type=int, default=0, help="Run batch with N synthetic images")
    parser.add_argument("--health", action="store_true", help="Just check health")
    args = parser.parse_args()

    if args.health:
        check_health(args.gateway)
        return

    check_health(args.gateway)

    if args.batch > 0:
        images = [create_demo_image() for _ in range(args.batch)]
        run_batch_inference(args.gateway, args.model_id, images)
        return

    if args.image:
        b64 = image_to_b64(args.image)
    elif args.demo:
        b64 = create_demo_image()
        print("[INFO] Using synthetic random image (demo mode)")
    else:
        print("[!] Provide --image <path> or --demo flag")
        parser.print_help()
        sys.exit(1)

    if args.async_mode:
        run_async_inference(args.gateway, args.model_id, b64)
    else:
        run_sync_inference(args.gateway, args.model_id, b64)


if __name__ == "__main__":
    main()
