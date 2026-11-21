"""The code equality the thread tests hold: one set of Stored codes at any BLAS thread count.

`ogld.genotypes.r` is the same arithmetic whatever BLAS runs it on, and a tile's product must
equal the same rows' products taken singly and in blocks of another size. A test that holds this
runs the matrix at each thread count, at two block sizes and a row at a time, and asserts every
code equals the first count's (ogld#90).
"""

from __future__ import annotations

from collections.abc import Sequence

import build_fixtures as made
import numpy as np
import testdata
from threadpoolctl import threadpool_limits

from ogld.format import REncoding


def assert_codes_identical(threads: Sequence[int], k: int, scale: int) -> None:
    """Assert the Test Panel's codes equal at each of `threads` BLAS threads, two block sizes
    and one row at a time, at the covariates `k` and the encoding `scale`."""
    from ogld.genotypes import covariates, prepare, r

    rows = np.flatnonzero(testdata.variants().passes_filters)
    calls = made.hard_calls()[rows].astype(np.float64)
    encoding = REncoding(scale)
    expected = None
    for count in threads:
        with threadpool_limits(limits=count, user_api="blas"):
            cov = covariates(
                list(testdata.pcs().names[:k]), testdata.pcs().values[:, :k], calls.shape[1]
            )
            for size in (127, 256):
                blocks = [
                    prepare(calls[i : i + size], cov, made.a1_signs()[rows[i : i + size]] > 0)
                    for i in range(0, len(rows), size)
                ]
                codes = np.block([[encoding.quantise(r(a, b)) for b in blocks] for a in blocks])
                assert codes.shape == (len(rows), len(rows))
                if expected is None:
                    expected = codes
                np.testing.assert_array_equal(codes, expected)
            whole = prepare(calls, cov, made.a1_signs()[rows] > 0)
            single = np.concatenate(
                [encoding.quantise(r(whole[i : i + 1], whole)) for i in range(len(rows))]
            )
            np.testing.assert_array_equal(single, expected)
