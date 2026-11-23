"""The read side of the array seam: zarr's runtime configuration and the read-only open.

zarr-python 3 keeps behaviours that decide how fast a Store Release reads, what
a build writes, and whether a forked build worker can finish, in process-wide
runtime configuration rather than in array metadata (ADR 0056). They are set
here, once, when this module is imported. It is the module every Store array is
opened through -- the Store Reader's own reads, and `opengwasdb.store.arrays`,
the array-creation seam, which imports it -- so no Store array can be read or
written without them. ADR 0056 records why each is set; its 23 Nov 2026 note
records that the configuration moved here from `opengwasdb/store/arrays.py`
with the read-only open (#464).

`open_group` is the one read-only open of a Store's Zarr group. Every write --
an array or group created, a group opened in any other mode -- goes through
`opengwasdb.store.arrays`, which also guards writes against consolidated
metadata (ADR 0056 §4); a read cannot change a release, so it needs no guard.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numcodecs
import zarr
from zarr.core.sync import sync as zarr_sync
from zarr.storage import LocalStore

# The module whose import switches Blosc's threads off.  ``import zarr`` loads
# it today; importing it by name makes sure it has run before the setting below
# turns them back on, should zarr ever load it lazily.  (`importlib` rather than
# an import statement so mypy does not follow into zarr's sources.)
importlib.import_module("zarr.codecs.blosc")

# ── zarr runtime configuration ───────────────────────────────────────────────
#
# zarr-python 3 keeps three behaviours in process-wide runtime config rather
# than in array metadata.  They are set here, once, in the module every Store
# array is created and opened through, so every array the package touches gets
# them and none can be opened without them.  ADR 0056 records why each is set,
# what was rejected and what it costs.

#: zarr 2 wrote every chunk, including one that is entirely the fill value: its
#: ``write_empty_chunks`` default was True.  zarr-python 3 defaults it to False,
#: which would silently drop those chunk files from a built store and change
#: the file set a release holds.  The setting is *runtime* config, not stored
#: metadata, so pinning it per array only covers the object `create_array`
#: returns -- every ``group[name]`` reopens with the default.  It is therefore
#: set once, here, the module that owns the Store's physical layout; every
#: array this package creates or reopens then writes empty chunks as zarr 2
#: did.  #247 must revisit this when builders move to Zarr v3 shards.
zarr.config.set({"array.write_empty_chunks": True})

#: Blosc's internal threads, back on (#244).  ``import zarr`` runs
#: ``numcodecs.blosc.use_threads = False`` for the whole process, and zarr 3
#: decodes on worker threads where numcodecs' adaptive default would say no
#: anyway, so every chunk decoded single-threaded: ~4.5 ms against ~0.6 ms with
#: Blosc's 8 threads, for a ``[1000, 1000]`` int16 chunk of OGS-00009
#: (``benchmarks/zarr3_blosc_decode.py``).  zarr 2.18 decoded on the main
#: thread with 8 Blosc threads; this restores that.
#:
#: On its own, under zarr's default pipeline, it is not a speed-up: decodes
#: issued concurrently from zarr's pool queue on numcodecs' lock (below), so
#: one Analysis genome-wide went from 76 s to 88 s and random lookups slowed by
#: 21-41%, while phewas went from 50 ms to 34 ms (medians of three fresh
#: processes).  It pays off with the one-worker fused pipeline below, which
#: decodes one chunk at a time with all of Blosc's threads.
#:
#: It is safe, from numcodecs 0.17 (the pinned floor) and its c-blosc 1.21.7:
#:
#: * threads: a threaded call uses Blosc's one global context, and numcodecs
#:   serialises every such call -- compress and decompress -- under a module
#:   ``threading.Lock``, with the GIL released inside it.  Concurrent decodes
#:   queue rather than race.  (0.16 did not take the lock on decompress.)
#: * forks: a forked process never uses the global context, because numcodecs
#:   compares the pid with the importing process's and runs single-threaded
#:   context functions in a child whatever ``use_threads`` says; it also
#:   re-creates its lock in the child, and c-blosc discards the inherited global
#:   context in its own ``pthread_atfork`` child handler.
#:
#: It is global because the flag is: numcodecs has no per-call or per-thread
#: form, so a "query-only" switch could only mean "on from the first query
#: onwards", a mode that depends on what a process did earlier.  The cost
#: falls on builds, and it is the one zarr 2.18 already had: a chunk of at
#: least two Blosc blocks (256 KiB uncompressed at zstd clevel 3, so every
#: ``[1000, 1000]`` Dense chunk) compressed in the parent writes its blocks in
#: completion order, so its bytes differ run to run while its decoded values
#: and its compressed size do not.  Compare built stores byte for byte under
#: ``BLOSC_NTHREADS=1``, as #243 did.  The SE plan reads compressed sizes only,
#: and forked workers stay single-threaded.
if not hasattr(numcodecs.blosc, "use_threads"):
    raise ImportError(
        "numcodecs.blosc no longer has `use_threads`; the seam cannot restore threaded "
        "Blosc decoding, and setting the old name would silently do nothing (#244)"
    )
numcodecs.blosc.use_threads = True

#: Every array reads and writes through zarr's ``FusedCodecPipeline`` (opt-in
#: from zarr 3.3), with one worker (#244).  The default ``BatchedCodecPipeline``
#: schedules each chunk's fetch and decode as separate event-loop tasks; the
#: fused one fetches, decodes and scatters a whole selection in one hop to a
#: worker thread.  On OGS-00009 with Blosc threads on it took one Analysis
#: genome-wide from 88 s to 24 s (zarr 2.18: 26 s), random lookups from 186 ms
#: and 1,051 ms to 103 ms and 575 ms, and phewas from 34 ms to 23 ms (medians
#: of three fresh processes each).
#:
#: ``max_workers = 1`` is the measured choice, and the fork-safe one:
#:
#: * With Blosc threads on, chunk decodes queue on numcodecs' lock anyway, so a
#:   pool of workers adds contention, not decode throughput: with its default
#:   pool (224 workers here) the same read took 41-46 s, and random lookups
#:   were 23-42% slower; only the one-window read (4,241,966 associations)
#:   gained, by ~7%.
#: * With more than one worker the pipeline keeps a module-level
#:   ``ThreadPoolExecutor``, and zarr 3.4's after-fork reset clears its event
#:   loop and executor but not that pool (zarr-developers/zarr-python#4478).
#:   A forked build worker inherits the pool without its threads.  A read there
#:   of more than one chunk, but of no more chunks than the idle permits the
#:   parent's pool left, queues work that nothing runs and never returns --
#:   reproduced with the package's own ``ordered_map`` and on OGS-00009's
#:   top-hit gather.  A single-chunk read never uses the pool, so one-chunk
#:   fixtures cannot show it.  With one worker the pool is never created; do
#:   not raise ``max_workers`` (ADR 0056).
#:
#: Writes take the same path: a band write encodes its chunks one at a time
#: (multi-threaded inside Blosc), as zarr 2.18 did.  A forked worker decodes
#: single-threaded and now also one chunk at a time.
_FUSED_PIPELINE = "zarr.core.codec_pipeline.FusedCodecPipeline"
if not hasattr(importlib.import_module("zarr.core.codec_pipeline"), "FusedCodecPipeline"):
    raise ImportError(
        f"{_FUSED_PIPELINE} is gone; the seam's read path and its fork guard were "
        "written for it (#244)"
    )
zarr.config.set({"codec_pipeline.path": _FUSED_PIPELINE, "codec_pipeline.max_workers": 1})


def array_length(array: Any) -> int:
    """The leading dimension of an array -- what zarr 2's ``len(array)`` meant.

    zarr-python 3 removed ``Array.__len__``, so ``len(array)`` now raises
    ``TypeError``; the leading axis is the length every call site wanted.  A
    2-D top-hit tier reads the same way as a 1-D CSR plane.
    """
    return int(array.shape[0])


def inner_chunk_of(array: Any) -> tuple[int, ...]:
    """The *inner* chunk of a Store array, sharded or not.

    zarr-python 3's ``Array.chunks`` is the inner chunk for a sharded array and
    ``Array.shards`` is the outer shard -- the reverse of what a reader might
    assume from the v2 spelling.  Every rule about the unit a query reads (the
    per-variant chunk rule, issue #135; the recorded-layout rule, #245) must
    use this, never ``shards``, or it would judge the file unit instead of the
    read unit.  A store with one chunk per shard has ``chunks == shards``.
    """
    return tuple(int(size) for size in array.chunks)


def open_group(path: str | Path) -> Any:
    """Open a Store Release's Zarr group read-only.

    The one place a reader calls `zarr.open_group`. The stored metadata decides
    the Zarr format, so a 0.1.0 (Zarr v2) release opens as a 0.2.x (v3) one
    does. Every write -- and every other open mode -- goes through
    `opengwasdb.store.arrays` (`open_group`, `open_group_for_write`).
    """
    store: LocalStore = zarr_sync(LocalStore.open(root=Path(path), mode="r", read_only=True))
    return zarr.open_group(store, mode="r", zarr_format=None)
