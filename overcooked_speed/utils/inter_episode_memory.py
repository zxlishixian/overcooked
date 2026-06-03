"""Cross-episode memory buffer for LTS-PPO.

Stores M most recent episode characteristic vectors c_{-i}^e.
Used by the InterEncoder to provide long-horizon teammate behavior context.
"""
import numpy as np


class InterEpisodeMemory:
    """Ring buffer storing M most recent episode characteristic vectors.

    Unfilled slots (when fewer than M episodes have been recorded) are zeros.
    get_memory() always returns a fixed-size flattened vector (M * c_dim,).
    """

    def __init__(self, M=10, c_dim=17):
        self.M = M
        self.c_dim = c_dim
        self._buffer = np.zeros((M, c_dim), dtype=np.float32)
        self._count = 0
        self._write_idx = 0

    def push(self, c):
        """Add a new episode characteristic vector."""
        self._buffer[self._write_idx] = np.asarray(c, dtype=np.float32)
        self._write_idx = (self._write_idx + 1) % self.M
        self._count = min(self._count + 1, self.M)

    def get_memory(self):
        """Return flattened memory in chronological order (oldest first).

        Returns:
            np.ndarray of shape (M * c_dim,) float32.
            Unfilled slots are zeros.
            When M=0, returns an empty array of shape (0,).
        """
        if self.M == 0:
            return np.zeros(0, dtype=np.float32)

        result = np.zeros(self.M * self.c_dim, dtype=np.float32)
        for i in range(self._count):
            src_idx = (self._write_idx - self._count + i) % self.M
            start = i * self.c_dim
            result[start:start + self.c_dim] = self._buffer[src_idx]
        return result

    def get_count(self):
        """Return number of episodes stored so far."""
        return self._count

    def reset(self):
        """Clear all stored data."""
        self._buffer.fill(0.0)
        self._count = 0
        self._write_idx = 0
