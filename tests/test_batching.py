from __future__ import annotations

from types import SimpleNamespace

import pytest

from main_flow.batching import experience_update_interval_samples


def test_experience_update_interval_samples() -> None:
    args = SimpleNamespace(experience_large_batch=8, rollouts_per_sample=4)
    assert experience_update_interval_samples(args) == 2


def test_experience_update_interval_requires_full_samples() -> None:
    args = SimpleNamespace(experience_large_batch=10, rollouts_per_sample=4)
    with pytest.raises(ValueError, match="must be a multiple"):
        experience_update_interval_samples(args)

