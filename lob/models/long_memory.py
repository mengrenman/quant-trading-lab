"""Long-memory buy/sell sign processes for order-flow models."""

from __future__ import annotations

import numpy as np


def farima_weights(d: float, n: int) -> np.ndarray:
    """First n moving-average weights of FARIMA(0, d, 0).

    psi_k = Gamma(k + d) / (Gamma(d) Gamma(k + 1)), computed by the recursion
    psi_k = psi_{k-1} (k - 1 + d) / k. They decay like k^(d - 1).
    """
    k = np.arange(1, n)
    return np.concatenate([[1.0], np.cumprod((k - 1 + d) / k)])


class LongMemorySigns:
    """Endless stream of +1/-1 signs with long memory.

    The signs are those of FARIMA(0, d, 0) noise with d = H - 1/2, whose
    autocorrelation decays like k^(2H - 2) (a Hurst exponent of H). The
    infinite moving average is truncated at ``memory`` lags and applied in
    chunks by overlap-save FFT convolution: each chunk reuses the previous
    chunk's innovations, so the series continues seamlessly and its memory
    never resets, unlike a generator that draws independent blocks.

    Args:
        hurst: Hurst exponent, 0.5 (no memory) <= H < 1.
        memory: Number of lags kept in the moving average; also the chunk size.
    """

    def __init__(self, hurst: float, memory: int = 2 ** 16):
        if not 0.5 <= hurst < 1:
            raise ValueError(f"hurst must be in [0.5, 1), got {hurst}")
        self.hurst = hurst
        self.memory = memory
        self._filter = np.fft.rfft(farima_weights(hurst - 0.5, memory), 2 * memory)
        self._tail: np.ndarray | None = None
        self._signs = np.empty(0, dtype=np.int8)
        self._idx = 0

    def _refill(self, rng: np.random.Generator) -> None:
        if self._tail is None:
            # Innovations before the first output, so the stream starts stationary.
            self._tail = rng.standard_normal(self.memory)
        fresh = rng.standard_normal(self.memory)
        segment = np.concatenate([self._tail, fresh])
        noise = np.fft.irfft(np.fft.rfft(segment) * self._filter, 2 * self.memory)[self.memory:]
        self._tail = fresh
        self._signs = np.where(noise >= 0, 1, -1).astype(np.int8)
        self._idx = 0

    def next(self, rng: np.random.Generator) -> int:
        """Return the next sign (+1 buy, -1 sell)."""
        if self._idx >= len(self._signs):
            self._refill(rng)
        sign = int(self._signs[self._idx])
        self._idx += 1
        return sign
