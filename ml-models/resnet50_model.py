"""
Sample PyTorch Model: ResNet50 Image Classifier
-------------------------------------------------
Demonstrates:
  - CUDA-accelerated inference
  - TorchScript compilation
  - Model saving/loading for the platform
  - Sample inference loop with benchmarking
"""

import time
import base64
import io
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ── Build model ───────────────────────────────────────────────────────────────
def build_resnet50(num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
    """
    Load a ResNet-50 with optional ImageNet pretrained weights.
    For a custom dataset, set pretrained=False and num_classes=<your count>.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if num_classes != 1000:
        # Replace final FC layer for fine-tuning
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ── Save model ────────────────────────────────────────────────────────────────
def save_model(model: nn.Module, path: str):
    """Save model weights (state_dict) to disk."""
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved → {path}")


def save_full_model(model: nn.Module, path: str):
    """Save entire model object (lets inference service load without class def)."""
    torch.save(model, path)
    print(f"[INFO] Full model saved → {path}")


def export_torchscript(model: nn.Module, path: str):
    """Export to TorchScript for optimised production inference."""
    model.eval()
    example = torch.randn(1, 3, 224, 224).to(DEVICE)
    model_ts = torch.jit.trace(model.to(DEVICE), example)
    model_ts.save(path)
    print(f"[INFO] TorchScript model saved → {path}")
    return model_ts


# ── Preprocessing ─────────────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(image_path: str) -> torch.Tensor:
    """Load an image from disk and return a preprocessed tensor."""
    img = Image.open(image_path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)  # add batch dim


def image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string (for API calls)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model: nn.Module, tensor: torch.Tensor) -> tuple:
    """Run inference and return (class_idx, confidence)."""
    tensor = tensor.to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5)
    return top5_idx[0].tolist(), top5_prob[0].tolist()


# ── Benchmark ─────────────────────────────────────────────────────────────────
def benchmark(model: nn.Module, n: int = 100, batch_size: int = 8):
    """Measure inference throughput."""
    model.eval().to(DEVICE)
    dummy = torch.randn(batch_size, 3, 224, 224).to(DEVICE)

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    total_images = n * batch_size
    throughput = total_images / elapsed

    print(f"\n{'='*50}")
    print(f"  Benchmark Results ({DEVICE})")
    print(f"{'='*50}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Iterations      : {n}")
    print(f"  Total images    : {total_images}")
    print(f"  Total time      : {elapsed:.2f}s")
    print(f"  Throughput      : {throughput:.1f} images/sec")
    print(f"  Avg latency     : {elapsed/n*1000:.2f}ms/batch")
    print(f"{'='*50}\n")

    return throughput


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ResNet50 Inference Demo")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    parser.add_argument("--save", type=str, default="resnet50.pt", help="Save model path")
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--export-ts", type=str, help="Export TorchScript to this path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("[INFO] Loading ResNet50 (ImageNet pretrained)...")
    model = build_resnet50(pretrained=True)
    model.eval().to(DEVICE)

    if args.save:
        save_full_model(model, args.save)

    if args.export_ts:
        export_torchscript(model, args.export_ts)

    if args.image:
        tensor = preprocess(args.image)
        top5_idx, top5_prob = infer(model, tensor)
        print("\nTop-5 Predictions:")
        for rank, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
            print(f"  {rank}. class_id={idx:<6}  confidence={prob:.4f}")

    if args.benchmark:
        benchmark(model, n=args.iterations, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
