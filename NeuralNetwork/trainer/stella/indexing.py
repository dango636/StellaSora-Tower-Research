"""Host-side integer contracts; never inspect CUDA tensor values here."""
from __future__ import annotations

from numbers import Integral


def require_integer(value, name, *, minimum=0, maximum=None):
    """Accept genuine integer scalars without truncation or wraparound."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}, got {result}")
    return result


def action_index(value, count, name):
    """Return a legal index, or None for the explicit unknown None/-1 sentinel.

    Python accepts q[-1] as the final action, while torch.gather rejects -1.
    Unknown archived decisions must therefore be excluded from ranking, never
    converted into a real last-action label or clamped to action zero.
    """
    if value is None:
        return None
    result = require_integer(value, name, minimum=-1, maximum=count-1)
    return None if result == -1 else result
