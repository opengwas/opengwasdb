"""Where an LD Store is: a local path or an HTTP(S) URL, opened as one Zarr hierarchy (spec §1).

A reader and validation open a store the same way, so a store served from a static host that
honours byte ranges reads exactly as it does from disk. zarr opens a URL through fsspec's
HTTP filesystem (aiohttp), and asks only for the byte ranges of the inner chunks a read
needs (spec §10). Nothing here lists a directory: every node is found by its own metadata
document, which a static host serves like any other file.
"""

from __future__ import annotations

import os

import zarr

from ogld.format import InputError


def open_group(location: str | os.PathLike[str]) -> zarr.Group:
    """The root group of the hierarchy at `location`, read-only, whatever its Zarr format.

    A location with no Zarr group at all is an `InputError` naming it. A Zarr v2 group opens,
    so validation can say it is not v3 (spec §12, `envelope.arrays`) rather than not there.
    """
    try:
        return zarr.open_group(str(location), mode="r")
    except FileNotFoundError as missing:
        raise InputError(f"no LD Store at {location}: there is no Zarr group there") from missing
