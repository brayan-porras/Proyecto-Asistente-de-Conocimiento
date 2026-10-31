import functools
import time
from collections.abc import Callable
from typing import Any

from app.domain.models import Step


def traced(agent: str, node: str) -> Callable:
    def decorator(fn: Callable[[Any, dict], dict]) -> Callable:
        @functools.wraps(fn)
        def wrapper(self: Any, state: dict) -> dict:
            started = time.perf_counter()
            update = fn(self, state)
            step = Step(
                agent=agent,
                node=node,
                ms=int((time.perf_counter() - started) * 1000),
                tokens=update.pop("_tokens", 0),
                detail=update.pop("_detail", ""),
            )
            update["steps"] = [step]
            return update

        return wrapper

    return decorator
