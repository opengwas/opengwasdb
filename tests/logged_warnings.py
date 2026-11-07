"""What the package logged while one call ran, in a test.

A Ragged completion logs a warning naming each Analysis whose SE exception share
breaches the budget (ADR 0078), and the test that reads those lines has to
attach a handler around the one call that emits them. The session fixture that
builds the completed release completes once too, and its completion is the one
an under-budget Analysis's silence is asserted on, so the handler lives here
rather than beside each caller.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def logged_warnings(logger: str = "opengwasdb") -> Iterator[list[str]]:
    """`with logged_warnings() as messages:` -- every warning logged inside it."""
    handler = _Warnings()
    target = logging.getLogger(logger)
    target.addHandler(handler)
    try:
        yield handler.messages
    finally:
        target.removeHandler(handler)
