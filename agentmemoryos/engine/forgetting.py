import math
import time


# based on Ebbinghaus forgetting curve (1885)
# R = e^(-t/S) where t=time elapsed, S=stability
# we add access_count as a stability booster — more recalls = slower forgetting

# stability multipliers per importance bucket
# higher importance = slower decay. these values are tuned empirically
STABILITY_TABLE = {
    "low":    1.0,
    "medium": 2.5,
    "high":   6.0,
}


def _importance_bucket(importance: float) -> str:
    if importance < 0.35:
        return "low"
    elif importance < 0.7:
        return "medium"
    return "high"


def compute_retention(
    importance: float,
    created_at: float,
    last_accessed: float,
    access_count: int,
    now: float = None
) -> float:
    """
    Returns a retention score between 0 and 1.
    1.0 = fully retained, 0.0 = forgotten.

    Stability grows with access_count (each recall strengthens the memory).
    """
    now = now or time.time()
    t_hours = (now - last_accessed) / 3600.0

    if t_hours <= 0:
        return 1.0

    bucket = _importance_bucket(importance)
    base_stability = STABILITY_TABLE[bucket]

    # each access roughly doubles stability up to a ceiling
    # not perfectly scientifically grounded but works well in practice
    stability = base_stability * (1 + math.log1p(access_count) * 0.8)

    retention = math.exp(-t_hours / (stability * 24))  # stability in days
    return round(max(0.0, min(1.0, retention)), 4)


def should_forget(retention: float, threshold=0.1) -> bool:
    return retention < threshold


def time_until_forgotten(
    importance: float,
    created_at: float,
    last_accessed: float,
    access_count: int,
    threshold=0.1
) -> float:
    """
    Returns estimated hours until retention drops below threshold.
    Returns -1 if already below threshold.
    """
    current = compute_retention(importance, created_at, last_accessed, access_count)
    if current <= threshold:
        return -1.0

    bucket = _importance_bucket(importance)
    base_stability = STABILITY_TABLE[bucket]
    stability = base_stability * (1 + math.log1p(access_count) * 0.8) * 24  # in hours

    # solve R = e^(-t/S) for t: t = -S * ln(threshold)
    hours = -stability * math.log(threshold)
    elapsed = (time.time() - last_accessed) / 3600.0
    remaining = hours - elapsed
    return max(0.0, round(remaining, 2))
