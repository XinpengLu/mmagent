"""Batch-size helpers for train-time memory updates."""

from __future__ import annotations


def experience_update_interval_samples(args) -> int:
    """Return how many completed samples trigger one experience update."""
    rollouts_per_sample = int(args.rollouts_per_sample)
    large_batch = int(args.experience_large_batch)
    if large_batch % rollouts_per_sample != 0:
        raise ValueError(
            f"experience_large_batch ({large_batch}) must be a multiple of "
            f"rollouts_per_sample ({rollouts_per_sample})"
        )
    return max(1, large_batch // rollouts_per_sample)

