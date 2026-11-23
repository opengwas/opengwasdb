# Local mmap ALIDs and whole clumping (#81)

Measured 19 November 2026, porting #47 (`9b6fdb6`) onto main `f65f000`. The current
schema-2, first-use measurements are at production-code commit `5a6600e`; the original
schema-1 measurements at `903824c` are retained below as historical evidence. ADR 0098
defines the representation; ADR 0097's Partner Rows, frame codec, thresholds and worker
default are unchanged. This is the mmap part of #81, not its remaining decode/gather work.

**Round-2 trust-model clarification:** these answer comparisons concern trusted ogld
publications, not forged generations. Both the writer and derivation now run the same bounded
exact canonical correspondence check as `alid_index.count` before publishing a new pointer.
At first use the reader checks integrity and every canonical file's content binding, not
correspondence or producer authenticity. A hand-forged generation with recomputed payload
hashes, manifest and pointer is outside reader detection (as with other forged derived arrays);
`ogld validate` detects its noncorrespondence. Canonical mutation after attachment is outside
the immutability assumption. This read-only publication check changes no published bytes;
the measurements below are retained, with no new re-timing or scratch re-derivation.

## Shared protocol and artifacts

All artifacts are under `/data/opengwasdb/work/orchestrate/ogld-81/results/`:

- `prepare81.py`, `*-derivation-exactness.json`, `derive-*.log` and `validate-*.log`:
  scratch-only derivation, canonical-file hashes, row equality, and complete CLI validation.
- `benchmark81.py` extends the existing `harness/realdata.py` without changing its oracle;
  `time81.sh` runs the matrix, `timing81.log` records it, and `e2e-*.json` holds every run.
- `plink81.py`, `plink2/inputs/`, `plink2/{cold,warm}-{1,2,3}/`: exact recipe reports,
  all PLINK logs/stdout/stderr/outputs, commands, residency and comparisons.
- `summary.json`, `timing-table.md` and `summarize81.py`: medians/ranges and checks of all
  36 ogld and six PLINK receipts. `execution.json` pins code/harness/input hashes and versions.
- `red-integrity.log`, `red-correspondence.log`, `red-close.log`: runtime-only mutation
  failures before the final unmutated green runs. No source mutation is left behind.

Each ogld cell is three fresh Python processes, alternating cold/warm within each repeat,
under `/data/opengwasdb/work/orchestrate/ogld-45/benchmark.lock`, `taskset -c 0-7`, and one
native numerical thread. Inputs are GCST90476460's 260,821 matched ALIDs with absolute z
scores, r² 0.01, kb 3000. Importing/loading inputs and preparing cache state are outside timers.

One first clump is timed from ALIDs: `ogld.open`, `rows_of`, `_components` (LD), and the
whole `clump` call. There is no preparatory lookup or clump on that reader. Only these coarse
seams and `loci_of` are instrumented, not each frame/row; phase timing therefore has much less
per-row profiling overhead than the earlier diagnostic harness. Whole clump also includes
score checks, hit deduplication, loci, ordering and result assembly. Open+clump includes setup
and instrumentation overhead between those calls; separate medians need not sum.

Cold evicts and independently verifies **zero resident pages** with mincore for root metadata,
every file under `variants/` (including the companion), and every Partner Rows file, metadata
and offsets included. Warm pre-reads all those files and verifies all pages resident. Read
traces assert no `near/` or `far/` read; neither is therefore prepared. Every receipt retains
per-file residency and the real Store keys read, affinity, `uptime` and post-run load. These are
point-in-time Linux page-cache observations, not hardware-cache guarantees. All 36 results
match the complete baseline fingerprint, not just lead count.

## Timings — schema 2, verify on first use

The review reproduced a foreign companion changing rows despite identical root/index metadata.
Schema 2 now hashes every canonical index file before trusting companion bytes. Each EUR
manifest has 29 source pins: normalized root metadata and 28 canonical array metadata/shard
files, about 66.7 MB, in addition to 307.7 MB of companion payloads. Verification, mapping
and sticky failure are once per reader under a lock shared with close. Source inventory also
follows shard-directory links: extending the foreign-companion breaker to linked shards was
observed failing before fixing the incomplete walk. Cycles and broken/nonregular sources
refuse, rather than silently disappearing from the pin set. Ordinary open and
row-based matrix/region reads do not hash the companion. ADR 0098 records the revised contract.

New artifacts are in `results/fix1/`: `prepare.py`, `benchmark.py`, `time.sh`, `summary.json`,
`execution.json`, `e2e-*.json`, `*-exactness.json`, `head-*-fingerprints.json`, `validate-*.log`
and `validation-times.log`. `benchmark.py` extends the original harness, separately timing
`_mapped_alids` inside the first `rows_of`; it does not pre-verify or warm a timed lookup.
Cold/warm preparation, flock, affinity, input and full-result oracle are the shared protocol
above. All 15 fresh-process answers match baseline. The companion-less k10 reference was
repeated in the same alternating session, not borrowed from the schema-1 session. All rows use two workers.

Seconds: median [minimum–maximum]. **Verify is included in ALID→rows, not added again.**

| Store | state | open | ALID→rows | LD | C | O+C | verify | load |
|---|---|---|---|---|---|---|---|---|
| 81-eur | cold | .022 [.019–.028] | 1.769 [1.148–1.859] | 2.677 [1.407–2.744] | 4.668 [2.678–4.840] | 4.691 [2.697–4.868] | .551 [.345–.597] | 4.20–9.27 |
| 81-eur | warm | .010 [.010–.015] | 1.061 [.909–1.066] | 1.022 [.960–1.058] | 2.128 [2.029–2.267] | 2.138 [2.039–2.282] | .273 [.272–.284] | 3.57–8.16 |
| 81-eur-k0 | cold | .018 [.013–.020] | 1.490 [1.394–1.569] | 1.621 [1.363–2.797] | 3.163 [3.016–4.566] | 3.181 [3.029–4.586] | .526 [.525–.560] | 3.44–7.99 |
| 81-eur-k0 | warm | .014 [.012–.015] | 1.099 [1.063–1.289] | 1.063 [1.037–1.490] | 2.466 [2.271–2.685] | 2.481 [2.284–2.699] | .273 [.273–.469] | 3.15–7.21 |
| 69-eur | cold | .017 [.016–.019] | 2.686 [1.710–2.849] | 1.947 [1.278–2.626] | 5.000 [3.101–5.492] | 5.016 [3.118–5.511] | — | 3.15–6.71 |

First-use integrity/attachment costs .551 s k10 / .526 s k0 cold by median: .121 / .096 s
above the old .43 s open cost, below the allowed +.3 s. Every individual check is below .73 s.
This compares the moved integrity cost with old opening; **whole first lookup** is separately
reported as 1.769 / 1.490 s, not claimed subsecond. Excluding verification, k10 lookup is
1.218 [.803–1.262] s cold and .777 [.637–.793] s warm. Opening alone falls to .022 / .018 s
cold. LD code is unchanged; variable cold LD and the same-load reference caution against
attributing cross-session LD differences to schema changes. #40's 10 s target is still met;
#81's under-1 s cold LD target is not met.

Both schema-1 root/manifests were archived before removal. Realpath checks precede scratch
writes and old-generation deletion, all under their own `scratch/64/81-*`. New derivation takes
54.44 / 56.97 s. Canonical Variant Set hashes are unchanged; neither `69-*` nor any published
Store/input was modified. With/without-companion lookup matches all 260,821 target rows and
119,632 chr22 rows on each Store; all 28 full fingerprints per Store match the companion-less baseline.
The final inventory walk retains identical generations/source pins, rechecks exact rows and
fingerprints, and complete `ogld validate --threads 8` passes 30/30 on both again
(25:58 k10 / 25:52 k0). The earlier schema-2 implementation's receipts are preserved in
`results/fix1/before-linked-inventory/`, not mixed into the final timing summary.

Four new breakers were observed failing before implementation: foreign companion, canonical
file-pin inventory, schema-1 refusal, and opening/row-based reads avoiding hashes. Runtime
mutations restoring metadata-only pins, eager attachment, retry after failed verification,
or unlocked close each fail their respective breaker; restored code is green. Thread tests
cover one attachment across simultaneous lookups, sticky failure after repair, close before
lookup and close racing first attachment. Diagnostics assert Store, generation and actual
manifest, payload or canonical file. Intentional independent test oracles are documented.

Back-to-back default tiers, same locked environment: base `f65f000` 1,552 / **24.20 s**, head
1,601 / **20.06 s**, longest calls 1.48 / 1.55 s. The base archive has its own detached f65f000
Git identity and import-path assertion; test helpers are not inadvertently loading head code.
Uptime before/after base: load 2.23/4.14/4.25 → 5.38/4.74/4.45; immediately following head:
5.38/4.74/4.45 → 6.89/5.13/4.58. These noisy observations do not establish a test-speed gain.
Final coverage is 23.17 s, 97%, longest call 1.37 s. Earlier loaded runs (including 2.20 s
maximum call and 31.37 s coverage total) are retained rather than hidden. Five realdata tests
pass in 297.78 s; an earlier 305.34 s tier-budget overrun is also retained. Lint, strict
typecheck and all 12 postflight gates pass, with additions-only public-API warnings and no
quality/hook/baseline edits.

## Timings — schema 1, verify on open (historical)

Seconds: median [minimum–maximum], three processes per cell. Load is the range of the
one-minute load at timed-run start. The Stores have identical Variant Sets; k10 uses ×511,
k0 ×32767. Comparisons are shared-host observations, not controlled attribution of LD changes
to mmap: only ALID resolution/opening changed.

| Store | workers | state | open | ALID→rows | LD | whole clump | open+clump | load |
|---|---:|---|---|---|---|---|---|---|
| 81-eur | 1 | cold | .433 [.253–.489] | .825 [.746–1.161] | 2.258 [2.173–3.115] | 3.156 [3.097–4.470] | 3.531 [3.409–4.959] | 3.22–4.00 |
| 81-eur | 1 | warm | .240 [.239–.406] | .816 [.796–.922] | 1.551 [1.547–1.648] | 2.511 [2.485–2.715] | 2.751 [2.724–3.121] | 3.04–3.92 |
| 81-eur | 2 | cold | .436 [.425–.457] | .925 [.874–.948] | 1.499 [1.258–2.089] | 2.517 [2.354–3.173] | 2.954 [2.811–3.598] | 3.03–3.84 |
| 81-eur | 2 | warm | .243 [.241–.407] | .782 [.733–.933] | 1.112 [1.021–1.154] | 1.955 [1.947–2.236] | 2.196 [2.190–2.643] | 3.26–3.82 |
| 81-eur | 4 | cold | .433 [.425–.433] | .941 [.933–.942] | 1.681 [1.647–1.681] | 2.757 [2.734–2.762] | 3.183 [3.167–3.195] | 3.24–4.36 |
| 81-eur | 4 | warm | .400 [.394–.405] | .847 [.745–.928] | 1.344 [1.324–1.473] | 2.312 [2.286–2.418] | 2.712 [2.680–2.824] | 3.36–4.44 |
| 69-eur | 2 | cold | .020 [.017–.021] | 2.401 [1.821–2.471] | 2.729 [1.519–3.019] | 5.313 [3.471–5.700] | 5.330 [3.492–5.721] | 3.49–4.81 |
| 69-eur | 2 | warm | .012 [.011–.015] | 1.742 [1.729–2.388] | 1.213 [1.039–1.302] | 3.153 [2.882–3.757] | 3.164 [2.897–3.768] | 3.71–4.99 |
| 81-eur-k0 | 2 | cold | .449 [.429–.456] | .915 [.904–.964] | 2.136 [1.852–2.256] | 3.142 [2.930–3.377] | 3.591 [3.359–3.834] | 3.58–4.99 |
| 81-eur-k0 | 2 | warm | .266 [.265–.397] | .801 [.790–.874] | 1.003 [.886–1.022] | 1.942 [1.776–2.030] | 2.207 [2.042–2.427] | 3.49–4.76 |
| 69-eur-k0 | 2 | cold | .021 [.017–.028] | 2.433 [1.800–2.514] | 2.133 [1.333–2.635] | 4.728 [3.246–5.358] | 4.749 [3.263–5.386] | 3.37–4.61 |
| 69-eur-k0 | 2 | warm | .014 [.013–.018] | 2.275 [1.618–2.384] | .972 [.957–1.181] | 3.346 [2.672–3.718] | 3.364 [2.686–3.732] | 3.36–4.28 |

At two workers k10 ALID resolution falls from 2.401 to .925 s cold, and 1.742 to .782 s warm.
It is now subsecond by median, and less than LD work, but opening hashes the companion and
adds .436 s cold. Open+clump improves from 5.330 to 2.954 s cold and 3.164 to 2.196 s warm.
#40's ≤10 s cold whole-clump target is met. #81's <1 s cold LD target is **not**: best k10
median is 1.499 s at two workers, with a best individual run of 1.258 s.

## PLINK2 (timed once, with schema 1)

PLINK v2.0.0-a.6.9LM, same prepared read-only `realdata/EUR-k0.plink2/chr<N>` genotypes.
The recipe's 22 reports were regenerated with its own `minus_log10_p` and verified
byte-identical to the original reports. Command per chromosome:

```sh
plink2 --pfile <realdata>/EUR-k0.plink2/chr<N> --clump <results>/plink2/inputs/chr<N>.hits \
  --clump-log10 input-only --clump-p-field LOG10P --clump-log10-p1 0 --clump-log10-p2 0 \
  --clump-r2 0.01 --clump-kb 3000 --clump-unphased --threads 8 --memory 4000 \
  --out <results>/plink2/<state>-<repeat>/chr<N>
```

The exact recipe flags are retained except its `--threads 1` becomes 8. Each whole-Analysis
run is 22 serial fresh PLINK processes, under the same flock and CPU affinity. Preparation
is outside timing; process startup and reading all chromosomes are included. Cold evicts and
mincore-verifies all pgen/pvar/psam and report files; warm pre-reads and verifies them.

Whole Analysis: cold **2.078 [1.813–2.645] s**, load 1.69–1.82; warm
**1.521 [1.503–2.336] s**, load 1.69–1.88. All six runs return exactly the same **3,819 leads**
as `81-eur-k0`; neither side has an extra lead. No acceptance-snapshot exception is needed.
PLINK is faster than k0 ogld open+clump (3.591 s cold, 2.207 s warm) in these runs, but ran
under lower shared-host load. Neither benchmark includes genotype conversion or Store derivation.

## Correctness and checks (schema 1)

Each scratch companion holds 8,546,809 entries (307,685,124 payload bytes). Derivation took
57.22 s k10 / 55.95 s k0. Resolved write targets, generation payloads and root pointer are
inside their own `scratch/64/81-*` directory. Existing canonical Variant Set files retain
all pre-derivation hashes. No `69-*`, published Store, original genotype or Analysis input
was modified. Complete `ogld validate --threads 8` passes all 30 rules on both scratch Stores,
including `alid_index.*` (24 min 37 s k10 / 24 min 27 s k0).

With/without-companion lookup on each scratch Store returns identical int64 rows for all
260,821 target ALIDs and all 119,632 chr22 ALIDs. The target includes 10 exact-width and
59 over-width names; chr22 includes 44 exact-width and 765 over-width names. All 28 frozen
full-result fingerprints match baseline on each Store. Missing/HTTP/older Stores, malformed,
Absent, duplicate and width-boundary requests are also covered by the default tests.

Disabling integrity attachment makes eight corruption breakers fail; disabling exact canonical
comparison makes three self-consistent-payload breakers fail, including a row permutation;
disabling release makes the new reader-close breaker fail. Restoring the implementation is
green. The committed Test Store remains unchanged and companion-less; regenerating a Store
now produces a companion. Generic recursive Zarr listings of newly written Stores emit the
inherited #47 warning about the plain-file `alid_mmap` directory; normal opening and complete
validation do not. This port retains #47's representation rather than changing ADR 0097.

Base default tier: 1,552 passed in 14.63 s. Final head: 1,584 passed in 28.22 s (longest call
1.86 s); coverage runs 21.29–23.46 s (final longest call 1.55 s), 97% coverage. The five realdata tests pass
in 278.10 s. Lint, strict typecheck and postflight gates pass; public-API warnings are additions
only. No baseline, quality configuration, hook or public signature was changed.
