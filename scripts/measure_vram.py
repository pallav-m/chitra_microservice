"""Measure peak GPU VRAM for concurrent separations, to size the per-model caps.

Run on the target GPU box (CUDA only):

    uv run python scripts/measure_vram.py --concurrency 4 --seconds 30

It loads the real Demucs model and runs N concurrent separations of a synthetic
clip, then prints torch.cuda.max_memory_allocated(). Use the result to set
DEMUCS_MAX_CONCURRENCY so the worst-case simultaneous mix fits the GPU's VRAM.
"""
import argparse
import threading

import numpy as np
import torch

from app.config import settings
from app.models.demucs_model import separate


def _synthetic(seconds: float, sr: int = 44100) -> tuple[torch.Tensor, int]:
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    tone = np.sin(2 * np.pi * 220 * t)
    return torch.from_numpy(np.stack([tone, tone])), sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=settings.demucs_max_concurrency)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    if settings.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit(f"CUDA required; resolved device is {settings.device!r}")

    waveform, sr = _synthetic(args.seconds)
    torch.cuda.reset_peak_memory_stats()

    def work():
        separate(waveform, sr)

    threads = [threading.Thread(target=work) for _ in range(args.concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    per_req = peak_gb / args.concurrency
    print(f"concurrency      : {args.concurrency}")
    print(f"clip length      : {args.seconds:.0f}s @ {sr} Hz")
    print(f"peak VRAM        : {peak_gb:.2f} GB")
    print(f"~per-request avg : {per_req:.2f} GB (includes shared weights amortized)")
    print("Set DEMUCS_MAX_CONCURRENCY so peak stays well under total VRAM.")


if __name__ == "__main__":
    main()
