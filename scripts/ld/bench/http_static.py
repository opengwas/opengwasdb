#!/usr/bin/env python3
"""A static host for LD Stores and a latency proxy in front of it, on this machine (ogld#8).

An LD Store is plain Zarr, so a static host that honours byte ranges serves it (ld-store §10). This
is that host, on loopback, and a proxy in front of it that waits a set time before forwarding
each request, to stand in for a network's round trip. It is loopback with injected latency, not
a real WAN: there is no bandwidth limit, no loss, and no extra round trip to open a connection,
which a real client pays once per connection (and more for TLS) and then reuses.

    python scripts/ld/bench/http_static.py static ROOT   # prints "ready PORT"
    python scripts/ld/bench/http_static.py proxy UPSTREAM  # prints "ready PORT"

The static host is aiohttp's `FileResponse`, which honours one byte range, a suffix range
included, and sends with sendfile. The proxy streams each response through and counts what
crosses it, per array: requests and body bytes. Its control paths are not counted:

- `GET /__ogld/stats` the counts as JSON; `POST /__ogld/reset` zeroes them;
- `POST /__ogld/latency/<ms>` sets the wait added to each request.

Each process runs one event loop and at most two worker threads, so a timing's thread budget
is the client's, plus these.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
from pathlib import Path

from aiohttp import ClientSession, TCPConnector, web

#: The proxy's client session, kept on its application.
SESSION = web.AppKey("session", ClientSession)
#: The headers a range response needs, passed through the proxy.
_PASSED = ("Content-Range", "Content-Type", "Accept-Ranges", "Last-Modified", "ETag")


def array_of(path: str) -> str:
    """The array a request is for, as `kind/name` (`near/r`, `index/partner`, `far/r`,
    `partners/frames`, `variants/alid`), or `metadata` for a `zarr.json` document and for the
    Zarr v2 documents (`.zgroup`, `.zattrs`, `.zmetadata`) zarr probes for when it opens a URL. A
    chunk's path is the array's, then `/c/...`."""
    last = path.rsplit("/", 1)[-1]
    if last == "zarr.json" or last.startswith("."):
        return "metadata"
    head = path.split("/c/", 1)[0].strip("/").split("/")
    # A store's own path under the served root comes first: keep the last component of the
    # group (`near`, `index`, `far`, `variants`), drop a chromosome, keep the array's name.
    if len(head) >= 3 and head[-3] in {"near", "index", "far", "partners"}:
        return f"{head[-3]}/{head[-1]}"
    return "/".join(head[-2:])


class Counts:
    """Requests and body bytes through the proxy, in all and per array."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.bytes = 0
        self.by_array: collections.Counter[str] = collections.Counter()
        self.requests_by_array: collections.Counter[str] = collections.Counter()

    def add(self, path: str, size: int) -> None:
        name = array_of(path)
        self.requests += 1
        self.bytes += size
        self.by_array[name] += size
        self.requests_by_array[name] += 1

    def as_json(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "bytes": self.bytes,
            "bytes_by_array": dict(self.by_array),
            "requests_by_array": dict(self.requests_by_array),
        }


def static_app(root: Path) -> web.Application:
    """Serve the files under `root` as they are, with byte ranges; no directory listing."""
    base = root.resolve()

    async def handle(request: web.Request) -> web.StreamResponse:
        path = (base / request.match_info["path"]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app = web.Application()
    app.router.add_get("/{path:.*}", handle)
    return app


def proxy_app(upstream: str) -> web.Application:
    """Forward every request to `upstream` after the set wait, streaming the body back."""
    counts = Counts()
    state = {"latency": 0.0}
    app = web.Application()

    async def start(app: web.Application) -> None:
        app[SESSION] = ClientSession(connector=TCPConnector(limit=0), auto_decompress=False)

    async def stop(app: web.Application) -> None:
        await app[SESSION].close()

    async def control(request: web.Request) -> web.Response:
        action = request.match_info["action"]
        if action == "reset":
            counts.reset()
        elif action.startswith("latency/"):
            state["latency"] = float(action.split("/", 1)[1]) / 1000
        return web.json_response({**counts.as_json(), "latency_ms": state["latency"] * 1000})

    async def forward(request: web.Request) -> web.StreamResponse:
        await asyncio.sleep(state["latency"])
        headers = {k: v for k, v in request.headers.items() if k in ("Range", "Accept")}
        session = request.app[SESSION]
        async with session.request(
            request.method, upstream + request.raw_path, headers=headers
        ) as r:
            out = web.StreamResponse(status=r.status)
            for key in _PASSED:
                if key in r.headers:
                    out.headers[key] = r.headers[key]
            out.content_length = r.content_length
            await out.prepare(request)
            size = 0
            async for piece in r.content.iter_chunked(1 << 20):
                size += len(piece)
                await out.write(piece)
            await out.write_eof()
        counts.add(request.path, size)
        return out

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_route("*", "/__ogld/{action:.*}", control)
    app.router.add_route("*", "/{path:.*}", forward)
    return app


async def started(app: web.Application) -> tuple[web.AppRunner, int]:
    """`app` running on a free loopback port: its runner, to clean up, and the port."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, int(runner.addresses[0][1])


async def _serve(app: web.Application) -> None:
    """Run `app` on a free loopback port, say which, and serve until killed."""
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=2)
    )
    _, port = await started(app)
    print(f"ready {port}", flush=True)
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("role", choices=["static", "proxy"])
    parser.add_argument("target", help="the directory to serve, or the upstream URL to proxy")
    args = parser.parse_args()
    if args.role == "static":
        app = static_app(Path(args.target))
    else:
        app = proxy_app(args.target.rstrip("/"))
    asyncio.run(_serve(app))


if __name__ == "__main__":
    main()
