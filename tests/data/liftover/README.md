# UCSC's hg19 → hg38 chain, for the test suite

`hg19ToHg38.over.chain.gz` is UCSC's chain file from
<https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz>.
It was last modified there on 1 January 2014. It is 227,698 bytes and holds 1,278 chains.

| | |
|---|---|
| md5 | `35887f73fe5e2231656504d1f6430900`, as UCSC's `md5sum.txt` gives it |
| sha256 | `5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d` |

The sha256 is the same as pyliftover's cached download (`~/.pyliftover`, #486). It is
also the same as the copy at `/data/opengwasdb/reference/liftover/`.

## Why it is here

pyliftover's built-in hg19 → hg38 (`LiftOver("hg19", "hg38")`) downloads the chain from
UCSC into `~/.pyliftover` the first time it runs. On 24 November 2026 that download failed
on GitHub's runner, and 200 tests failed and 147 more errored. So the test suite now never
downloads it. A session fixture in `tests/conftest.py`
(`_builtin_liftover_reads_the_committed_chain`) makes the built-in lift open this file,
with no cache and no web. A build pair with no chain here finds no file and fails loudly.
`tests/test_liftover_is_hermetic.py` lifts with the network blocked.

Production is unchanged. A real build reads pyliftover's own cache, or downloads the chain
there, or reads the `chain_file` it is given.

## Licence

This file is not covered by this repository's GPL-3.0 licence. It is UCSC's, on the terms
in `README.txt` beside it, which is UCSC's README for the directory, copied verbatim. In
short:

- it is free for non-commercial use by independent researchers and nonprofit
  organisations;
- any other use needs a commercial licence from UCSC;
- whoever redistributes it must include that README in the same directory, which is why
  `README.txt` is here.

The package's sdist and wheel ship only `opengwasdb/`, so neither carries this file.
