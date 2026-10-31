import functools
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from app.domain.models import Step

step_listener: ContextVar[Callable[[Step], None] | None] = ContextVar(
    "step_listener", default=None
)


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
            listener = step_listener.get()
            if listener:
                listener(step)
            return update

        return wrapper

    return decorator
