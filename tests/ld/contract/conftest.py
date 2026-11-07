"""Fixtures the contract tests share."""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from functools import partial

import pytest
from support import TEST_STORE, RangeHandler


@pytest.fixture
def served_store() -> Iterator[str]:
    """The Test Store's directory served over HTTP from this machine, as a URL."""
    handler = partial(RangeHandler, directory=str(TEST_STORE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join()
