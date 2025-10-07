"""Utilities for applying retry policies with exponential backoff and jitter."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class RetryableOperationError(Exception):
    """Exception used to signal retryable failures inside retry callbacks."""

    message: str
    error_code: Any | None = None
    retryable: bool = True

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _resolve_retryability(
    *,
    retryable: bool,
    error_code: Any | None,
    retryable_error_codes: Optional[Iterable[Any]],
) -> bool:
    if retryable_error_codes is None or error_code is None:
        return retryable
    try:
        return error_code in set(retryable_error_codes)
    except TypeError:
        # error_code may not be hashable; fall back to direct comparison
        return any(error_code == candidate for candidate in retryable_error_codes)


def retry_with_backoff(
    operation: Callable[[int, int], T],
    *,
    max_attempts: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter_factor: float = 0.25,
    operation_id: str,
    logger: logging.Logger,
    retryable_error_codes: Optional[Iterable[Any]] = None,
) -> T:
    """Execute ``operation`` applying exponential backoff with jitter."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info(
            "Operation %s starting attempt %s/%s", operation_id, attempt, max_attempts
        )
        try:
            return operation(attempt, max_attempts)
        except RetryableOperationError as exc:
            is_retryable = _resolve_retryability(
                retryable=exc.retryable,
                error_code=exc.error_code,
                retryable_error_codes=retryable_error_codes,
            )
            last_error = exc
            if not is_retryable:
                logger.error(
                    "Operation %s failed with non-retryable error on attempt %s/%s: %s",
                    operation_id,
                    attempt,
                    max_attempts,
                    exc,
                )
                raise
        except Exception as exc:  # pragma: no cover - defensive path
            last_error = exc
            logger.exception(
                "Operation %s raised unexpected exception on attempt %s/%s",
                operation_id,
                attempt,
                max_attempts,
            )
            raise

        if attempt == max_attempts:
            break

        sleep_time = min(max_delay, base_delay * (2 ** (attempt - 1)))
        if jitter_factor > 0 and sleep_time > 0:
            jitter = random.uniform(0, sleep_time * jitter_factor)
            sleep_time = min(max_delay, sleep_time + jitter)
        logger.info(
            "Operation %s will retry in %.2f seconds (attempt %s/%s)",
            operation_id,
            sleep_time,
            attempt,
            max_attempts,
        )
        time.sleep(sleep_time)

    assert last_error is not None
    logger.error(
        "Operation %s exhausted all %s attempts without success.",
        operation_id,
        max_attempts,
    )
    raise last_error
