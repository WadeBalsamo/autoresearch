"""Fixed-time-budget helper + VRAM tracking (torch optional).

Karpathy's autoresearch makes experiments comparable by giving each a fixed wall-clock
budget. We keep that: classification/NSP default to 5 min, the generative QLoRA track to
a larger budget (signal needs more steps). The budget excludes model download/compile —
start the clock after setup.
"""
from __future__ import annotations

import time
from typing import Optional


class Budget:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.t0: Optional[float] = None

    def start(self) -> "Budget":
        self.t0 = time.time()
        return self

    @property
    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.time() - self.t0

    @property
    def expired(self) -> bool:
        return self.t0 is not None and self.elapsed > self.seconds

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)


def peak_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return 0.0


def reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
