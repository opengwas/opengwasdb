# Direct decode and chromosome-local clumping (ogld#40)

## Measure first

At `f5e4845`, GCST90476460, 260,821 hits, r² 0.01, 3 Mb, the published EUR store:

| cold local run | seconds | all chunk fetches / distinct | Near Band fetches / distinct | leads |
|---|---:|---:|---:|---:|
| tip, zarr helper pool 1 | 137.220 | 51,641 / 35,474 | 49,561 / 33,394 | 3,869 |
| disposable prototype | 72.719 | 33,914 / 33,914 | 33,394 / 33,394 | 3,869 |

The prototype reads byte ranges through StorePath, verifies the index CRC with numcodecs,
decodes on five workers, visits chromosomes serially and omits unused variant columns.
It was measured before the production implementation. Both fingerprints (dtype and bytes
of all six result columns) were
`085516239b15dca279de9dd6869e7b275fddf62a3e1661d6fa615d315f4f2c36`.
The target is not met by this prototype; its remaining costs require measurement, not a
format change assumed to fix them.

Pages were advised away with `latency.evict`; opening is inside the timing. Shared-file
pages can be warmed by other processes. BLAS 1, zarr helper 1, five decode workers: eight
client threads including the caller and event loop. Full `uptime` beside each timing:

```
Tip before: 22:15:28 up 9 days, 3:38, 1 user, load average: 1.31, 1.33, 1.22
Tip after:  22:17:45 up 9 days, 3:40, 1 user, load average: 2.65, 1.86, 1.43
Probe before: 22:18:26 up 9 days, 3:41, 1 user, load average: 2.38, 1.88, 1.46
Probe after:  22:19:39 up 9 days, 3:42, 1 user, load average: 2.58, 2.06, 1.55
```

## Decoder evidence

Every Test Store array equals zarr, and a re-sharded numeric array with a padded last
chunk equals it in both endian orders. Tests cover an empty inner chunk, corrupt frame,
short decoded frame, corrupt CRC and truncated index. Four unsupported chains fall back
to zarr. Every Test Store array also equals local zarr through the contract HTTP server.
Wrong endianness failed the big-endian comparison; shifting the byte offset by one failed
both endian comparisons with invalid zstd input. Decoding a short frame directly into a
larger destination was observed to pass silently in numcodecs; the regression test failed
before the decoder was changed to check the decoded length.

#32's HTTP flight and held-buffer tests run on the direct path. The planner now reserves
half the decoded-byte allowance for worker scratch; the cache budget is unchanged.
The complete default tier passed 1,240 tests in 26.76 s with eight pytest workers at this
slice. Tip: 1,227 tests in 16.77 s, load 1.47→4.30. Both are below 30 s.
