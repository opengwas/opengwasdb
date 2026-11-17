# Epic #307: OGS-00009 after measurement (#322)

The release was built **once**, on 13 November 2026 at `8acd22b`: 2,514
Analyses, GWAS-VCF sources, 9,846,427 observed variants and 11,133,427 variants
after Reference Completion. The measured reader path is still per-row GWAS-VCF:
#316's integer join applies, but #318/#319's FinnGen/GWAS-SSF batches do not.

[The after-artifact](opengwasdb_epic307_after.json) derives each phase, its
October baseline and every epic projection from records. The baseline includes
backup contention; these are not matched-load before/after trials. Backups were
reported to have stopped covering this filesystem on 2 November; the recorded
13 November run showed no corresponding I/O slowdown. Its duration would not
fit between the old 15-hour backup windows, even had those still applied.

## The evening criterion is missed

The successful run spans **02:01:00.470425–12:13:58.044016 UTC**:
**36,777.574 s (10h12m57.574s)**, **4,377.574 s (1h12m57.574s) above nine hours**.
The earlier failed argument-check launch is excluded. No publication or
registration was performed.

| Step | Seconds |
|---|---:|
| Build, inline Top-Hit Index, 32 workers | 10,431.309 |
| Observed hm3 subset / overview | 461.978 / 3.220 |
| Observed validation, 32 workers | 867.871 |
| Reference Completion, 32 workers | 22,605.599 |
| Completed hm3 subset / overview | 469.687 / 2.327 |
| Completed validation, 32 workers | 989.389 |
| Observed-View Check | 945.868 |

Completion's timer summary attributes **16,443.2 s** to LD-block fits,
**4,060.9 s** to the Top-Hit Index, **1,149.1 s** to band writing, **675.5 s**
to checkpoint merge and **215.6 s** to axis construction. These summaries have
finer resolution than the integer-second log spans in the after-artifact.
The unchanged LD-block phase is **44.7% of the release** and the principal
milestone miss. Cheaper fits remain #423's Reference Completion Method question.
At that revision completion did not forward its worker count to the Top-Hit
Index scan: that phase was single-process despite `complete-dense --n-workers 32`.
Issue #435 now forwards it; the separate completed-Store measurement below does
not replace this historical whole-release result.

Every phase projection is met at the harness's stated 5%/one-second tolerance
except Pass 2: **34m55s against approximately 30 minutes**, still using per-row
GWAS-VCF. The epic's narrower six-step sum is **9h49m16s**, meeting its 19.4,
18.2 and 16.3-hour scenarios. That sum omits steps and is **not** milestone #388's
end-to-end result. The corresponding October six-step sum is **44h02m15s**,
not the epic's partly extrapolated approximately 27 hours.

## Correctness and the full-size #312 proof

Both 32-worker validation records report `ok: true`, no errors, findings or
warnings. The completed Store's Observed-View Check reports `passed: true`.
No single-worker re-validation was run.

The separately authorised 15 November standalone Top-Hit Index rebuild took
**158.053 s at 32 workers**. [Its comparison](epic307_322/top-hits-comparison.json)
finds **all 24 arrays exactly equal**, including every group/tier attribute.
All tiers span **three 1,024-Analysis column bands**:

| Tier | Hits | Hits in the three bands |
|---|---:|---|
| 5e-4 | 41,910,434 | 17,140,593 / 16,979,702 / 7,790,139 |
| 5e-6 | 11,020,928 | 4,715,139 / 4,185,164 / 2,120,625 |
| 5e-8 | 5,491,751 | 2,424,406 / 2,030,242 / 1,037,103 |

The read-only harness is now committed as `benchmarks/compare_top_hits.py`,
recovered from `scratch/322/run/compare_top_hits.py`.
[Its provenance note](epic307_322/top-hits-comparison-provenance.json) pins the
original result and both harness sources by SHA-256 and records the exact
reproduction invocation. The original full-size JSON and its numbers were not
rewritten or remeasured for this review. From this worktree, using only the two
already-built Stores:

```bash
pixi run -e dev python benchmarks/compare_top_hits.py \
  /data/opengwasdb/work/orchestrate/opengwasdb-307/scratch/322/OGS-00009/store.opengwasdb \
  /data/opengwasdb/work/orchestrate/opengwasdb-307/scratch/322/offclock/tophits/store.opengwasdb \
  --output /data/opengwasdb/work/orchestrate/opengwasdb-307/scratch/322/fix1/top-hits-reproduced.json
```

This command is documented, **not executed again** in the review round. It
compares the decoded index arrays and group/tier attributes and returns the
three-band counts above; it cannot write its report inside either input Store.

This proof is outside the timed release; it is not a second Store build. Its
scratch copy occupied 64 GiB, more than the authorisation's estimated 47 GiB.
The off-clock guard sampled available memory and free disk throughout copying
and both comparisons, with 200 GiB / 1 TiB stop thresholds. The standalone
step's sampled private-memory peak was 12.0 GiB, minimum available 916 GiB.

The [published comparison](epic307_322/observed-vs-published.json) completed once,
read-only, in **412.60 s**. [The attribution](epic307_322/comparison-attribution.json)
reads its outputs, Admission Record and eight variant metadata rows, not the
statistic planes. It accounts for the differences as follows:

- **#376:** 1,271 published-only rows, all off-primary contig lifts. No
  rebuilt-only rows; all 9,846,427 remaining rows match by ALID in the same order.
- **Engine #357's conflicting-duplicate rule (ADR 0072):** 15,226 published-only
  associations across eight variants. The Admission Record logs exactly
  **30,452 rejected source rows, two per affected Analysis/variant**, matching
  every absence count. These include the three collisions described by #325,
  but this measurement does not declare that still-open issue resolved.
- **Z:** zero differing values where both releases retain an association.
- **#355 and the residual refit:** 14,079,650,231 SE values differ: 5,002,503
  are direct float16 roundings; 13,788,253,543 are within the two codecs' bound;
  **286,394,185 exceed that codec-only bound**. The old Dense build narrowed
  *every* SE before fitting, not just exceptions. Adding its historical
  float16 log half-step gives **0.004425169954**, above the observed maximum
  absolute log ratio **0.004152023438** (codec-only bound **0.003937007874**).
  These counts are retained unchanged, not relabelled or hidden by loosening
  the comparator. The envelope accounts for the historical representation;
  it is **not an exact source-fidelity comparison**.
- **EAF:** 1,544 differing retained values, exclusively at four of the conflicting
  variants. Removing the conflicting observations changes all four EAF Baselines
  and hence residual quantisation; no other variant has an EAF value difference.
- **#381:** format 0.2.2 is used, but this pair needs **zero respelled ALIDs**.
  A format move does not imply this particular source changes allele order.

The aligned comparator now refuses different Reference Assemblies and permits
opposite-order historical indel pairing only across pre-0.2.2 sorted versus
0.2.2+ reference-first semantics. Opposite-order indels in two reference-first
Stores remain distinct insertion/deletion identities. This pair declares
GRCh38 on both sides and has zero respelled rows, so the original evidence and
strict SE counts remain unchanged; the full-size comparison was not repeated.

No additional difference class remains unexplained. The full source-by-source
fidelity of every retained cell was not remeasured; clean validation and the
bounded release comparison should not be mistaken for that stronger proof.

## Completion scan worker forwarding (#435)

On **16 November 2026**, at committed `8954c7fa`, the existing production
`build-dense-top-hits` command rebuilt the completed OGS-00009 index on a
**private reflink**, never on #322's authoritative completed Store. Dense
completion and Hybrid's initial and post-crossover scans now pass their existing
worker request to this same implementation. The public default remains one;
Ragged harvesting and #423's fit method are unchanged.

The [measurement record](epic307_435/measured-result.json) preserves argv,
commit, UTC timestamps, start/end `uptime`, memory planning and artifact hashes.
The run held `$RD/timing.lock`, **08:06:37.434361–08:10:13.705053 UTC**. The
[production log](epic307_435/rebuild.log) reports **32 requested and 32 effective
workers**, not an uncapped promise that every host gets 32. The unchanged cap
allowed a **236,522 MiB** budget, a quarter of available memory at pool creation.
The grid was **11,133,427 × 2,514**, scanned in **2,784 bands of 4,000 rows**,
under the unchanged **1 GiB decoded-band budget**.

| Phase | Seconds |
|---|---:|
| Scan | 191.1 |
| Tier write | 22.3 |
| Production CLI wall time (`time -v`) | 216.23 |

The phases are rounded to tenths by the production logger. The
[`time -v` record](epic307_435/rebuild-time-v.txt) includes a maximum RSS of
7,048,836 KiB; that is **not aggregate pool memory**. Before starting, the
conservative estimate was **102,579,642,368 bytes**, including worker units,
held results and a 16 GiB parent/writer allowance, below the quarter-available
**248,305,688,576 bytes**. [Ten-second `free -g` samples](epic307_435/free-g.log)
never fell below **910 GiB available**; a 200 GiB guard remained armed.

The [committed comparator's result](epic307_435/top-hits-comparison.json) finds
**all 27 arrays exactly equal**, with all group/tier attributes equal. This
completed index has the observed index's 24 arrays **plus three `imputed`
arrays**, one per tier. Hit counts are **47,805,439 / 12,750,070 / 6,405,322**
at `5e-4 / 5e-6 / 5e-8`, each spanning three Analysis column bands.
`analyses.tsv` is byte-identical, including every
[`n_hits_*` value](epic307_435/n-hits-columns.json) (SHA-256
`66dfa6cc67dfde289198061417bf3b5dde0fd33a43bf79be15bd963270200138`).
The source's 564-file manifest/Analytical Metadata/index hash inventory is
unchanged before and after; the inventories and guard driver remain under
`$RD/scratch/435/`.

This is **one standalone rebuild**, not a second whole Store build or Reference
Completion. The earlier 4,060.9-second completion phase was measured on another
date, not a matched-load serial control. No new end-to-end time is claimed,
and the unwaived nine-hour decision remains separate. Reproduce the read-only
comparison, not the rebuild, from this worktree with:

```bash
RD=/data/opengwasdb/work/orchestrate/opengwasdb-307
pixi run -e dev python benchmarks/compare_top_hits.py \
  "$RD/scratch/322/OGS-00009/completed.opengwasdb" \
  "$RD/scratch/435/completed.opengwasdb" \
  --output "$RD/scratch/435/top-hits-reproduced.json"
```

## Four reader rows left open by #311

The after-artifact carries #311's reader A/B **unchanged**, with its original
revision, timestamp and file hash under `reader_evidence_origin`. It does not
pretend that the old `reader_head` samples measured this branch. The separate
13 November cache probes are retained as raw logs under
[`epic307_322/reader-cache/`](epic307_322/reader-cache/), with row counts and
per-sample loads (approximately 46–81); they are exploratory, not a paired
revision A/B or a quiet-node trial.

| Row | #311 / epic | Probe evidence and attribution |
|---|---|---|
| `reader_ukb_floor` | 6.3446 / 8.2 s | Cold floor 7.9368–9.0072 s, median 8.7707; warm median 6.3066. The epic falls inside the cold range: consistent with cache state, not proof of its historical cause. |
| `reader_finngen_floor` | 22.2612 / 26.4 s | Cold floor 22.3053–27.3513 s, median 23.6366; warm median 22.374. Again consistent with cache state; causality remains unresolved. |
| `reader_ukb_control_over_floor` | 2.2627 / 1.9634 | Probe cold ratio 2.0582, warm 2.5346. Cache state moves the ratio in the required direction, toward the epic, but historical cache/load were not recorded. |
| `reader_finngen_control_over_floor` | 9.0057 / 7.7841 | Probe cold ratio 9.4271, warm 9.5013. The cache effect is much smaller than the gap from the epic. These probes cannot attribute the remainder: the epic recorded one sample and no host load/cache state. |

Each reason is checked against the row's own evidence. In particular, the last
row honestly names missing historical evidence instead of inventing a cause.

## Profiles and language decision

[The profile artifact](opengwasdb_epic307_profiles.json) summarises all recorded
whole-tree snapshots and three native worker recordings, with input hashes.
Native **790 pooled active samples** put 53.9% in OpenBLAS/numpy C/scikit-learn
Cython, 28.6% in Python frames and 17.5% in other native code. The draft's 70%
was the first recording alone and is superseded by this pooled result.
Non-native frames around numpy operations are not evidence of interpreter time.
The eight reader frames named by ADR 0081 account for **90,160 of 114,751**
Pass 2 worker samples across two snapshots: **78.57% pooled**, approximately
79%, correcting the draft's 87% for that same frame set.

[ADR 0081](../adr/0081-no-second-implementation-language-after-epic-307.md)
records the [owner's acceptance on 16 November 2026](https://github.com/opengwas/opengwasdb/issues/322#issuecomment-6904046834):
no compiled extension after #307; retain Python/Numba. Reconsider only if profiling
after the columnar GWAS-VCF reader and column-major band-write work identifies a
dominant interpreted per-row kernel that cannot be vectorised. The follow-ups
are to vectorise the remaining GWAS-VCF reader, benchmark column-major Dense
bands, and budget the SE rewrite pool; #435's completion index worker forwarding
is measured above. The SE rewrite's sampled
323.5 GiB private peak exceeded the initial approximately 233 GiB plan;
available memory never fell below 625 GiB. No new solver or numerical method
was introduced here.

## Reproduction from existing artifacts only

Let `RD=/data/opengwasdb/work/orchestrate/opengwasdb-307`. From this worktree:

```bash
pixi run -e dev python benchmarks/measure_epic307_evidence.py \
  --store OGS-00009=/data/opengwasdb/stores/OGS-00009 \
  --store OGS-00009-322="$RD/scratch/322/OGS-00009" \
  --window 2026-10-22T01:06/2026-10-22T16:10 \
  --daily-window 01:06-16:10 --tz Europe/London \
  --daily-from 2026-10-14 --until 2026-11-02 \
  --after OGS-00009-322 \
  --reader-evidence docs/benchmark-output/opengwasdb_epic307_evidence.json \
  --reader-cache-probes docs/benchmark-output/epic307_322/reader-cache \
  --output "$RD/scratch/322/after-reproduced.json"
pixi run -e dev python benchmarks/summarise_epic307_profiles.py \
  "$RD/scratch/322/run/e2e" --output "$RD/scratch/322/profiles-reproduced.json"
pixi run -e dev python benchmarks/attribute_epic307_comparison.py \
  "$RD/scratch/322" --output "$RD/scratch/322/attribution-reproduced.json"
```

The committed after-artifact additionally names #311's other seven baseline
Stores; its exact command is in `argv`. The raw profiles and full `.codes.npz`
comparison remain in scratch. **Do not rerun the Store build, completion,
standalone index rebuild or published comparison to regenerate these summaries.**
