import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from agentmemoryos.engine.forgetting import compute_retention, should_forget, time_until_forgotten


def test_fresh_memory_full_retention():
    now = time.time()
    r = compute_retention(
        importance=0.5,
        created_at=now,
        last_accessed=now,
        access_count=0,
        now=now
    )
    assert r == 1.0


def test_high_importance_decays_slower():
    now = time.time()
    last_accessed = now - (48 * 3600)  # 2 days ago

    low_r = compute_retention(0.1, last_accessed, last_accessed, 0, now)
    high_r = compute_retention(0.9, last_accessed, last_accessed, 0, now)

    assert high_r > low_r


def test_frequent_access_slows_decay():
    now = time.time()
    last_accessed = now - (72 * 3600)  # 3 days ago

    r_zero = compute_retention(0.5, last_accessed, last_accessed, 0, now)
    r_many = compute_retention(0.5, last_accessed, last_accessed, 20, now)

    assert r_many > r_zero


def test_should_forget():
    assert should_forget(0.05) is True
    assert should_forget(0.5) is False
    assert should_forget(0.09, threshold=0.1) is True  # just below threshold
    assert should_forget(0.1, threshold=0.1) is False  # threshold is exclusive (uses <)
    assert should_forget(0.11, threshold=0.1) is False


def test_retention_bounds():
    now = time.time()
    r = compute_retention(0.5, now, now, 0, now)
    assert 0.0 <= r <= 1.0

    # very old memory
    ancient = now - (365 * 24 * 3600)
    r_old = compute_retention(0.3, ancient, ancient, 0, now)
    assert 0.0 <= r_old <= 1.0


def test_time_until_forgotten_already_decayed():
    now = time.time()
    very_old = now - (1000 * 24 * 3600)
    result = time_until_forgotten(0.1, very_old, very_old, 0)
    assert result == -1.0


def test_time_until_forgotten_fresh():
    now = time.time()
    result = time_until_forgotten(0.5, now, now, 0)
    assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
