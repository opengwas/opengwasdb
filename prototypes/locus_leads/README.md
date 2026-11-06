# Locus Leads against plink2 clumping (throwaway prototype)

Epic #62 proposes one **Locus Lead** per locus, chosen by genetic distance: take
an Analysis's Top-Hit Query rows in descending `abs_z` (ties by ascending Store
Variant Index), and accept a row unless an accepted lead lies within W bp and
D cM. This prototype measures that rule against plink2 LD clumping on the
HGDP+1kGP genotype panels before anything is built into a Store. It is not part
of the package and will be deleted once the epic's decisions are made.

Run from the repository root:

```bash
# once: average the pyrho EUR population maps into one file
pixi run -e dev python -m prototypes.locus_leads build-map \
    --archive /data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/original/hg38_maps.tar.gz \
    --output /data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/EUR.tsv.gz

# one Analysis, one (D, W), every r^2
pixi run -e dev python -m prototypes.locus_leads compare \
    --store /data/opengwasdb/stores/OGS-00003/store.opengwasdb \
    --analysis finngen-r13-BMI_IRN \
    --genetic-map /data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/EUR.tsv.gz \
    --panel /data/opengwasdb/reference/hgdp1kgp-hg38/pgen/EUR \
    --d-cm 0.5 --w-bp 1000000 --r2 0.01 --r2 0.001 \
    --work-dir <scratch directory for plink2's files> \
    --output docs/benchmark-output/locus_leads/OGS-00003_finngen-r13-BMI_IRN.json
```

The Store, the map and the panel must be on one reference assembly, or the run
stops. The comparison uses Observed Associations only, and only the rows both
methods can see: matched to the panel by chromosome, position and allele set,
and with finite cM. Every other row falls in exactly one category under the
reason it was left out, and the counts must add up to the tier. plink2
is given -log10 p computed from z, with equal values separated in the rule's
order, so the two methods rank rows identically; a clump whose index variant
comes after one of its members in that order fails the run.

| module | holds |
|---|---|
| `rule.py` | the Locus Lead rule |
| `genetic_map.py` | interpolation, the map file and its identity, the archive check |
| `eur_map.py` | the EUR map, averaged from the pyrho population maps |
| `panel.py` | the genotype panel, and matching Store rows to it |
| `clump.py` | -log10 p, tie separation, plink2 clumping |
| `compare.py` | spurious splits and lost signals |
| `run.py` | one run, end to end; clumping is independent of D and W, so a sweep reuses it |
| `select.py` | the seeded, stratified draw of Analyses, and its record |
| `sweep.py` | every (D, W) and the bp-only baseline over the drawn Analyses; pooled rates and the working criteria |
| `pairs.py` | each disagreeing pair's r², distance band and spike, chance LD between chromosomes, and the pair file |
| `counts.py` | the count criteria C1-C3: rank, density against HapMap3, cM against bp-only |
| `ld.py` | r² between panel variants, from dosages plink2 exports |
| `profile.py` | the prominence rule: locus boundaries from valleys in the 5e-4 tier's \|z\| envelope (#409) |
| `blocks.py` | one lead per LD block, half-open, rows outside blocks counted (#409) |
| `profile_sweep.py` | both rules on #403's Analyses, shared set (checked row for row) and clumps (#409) |
| `profile_judge.py` | the rules' C1-C3 against bp-only at each W, and per-lead rates beside #403's (#409) |
| `profile_valleys.py` | a seeded sample of no-row valleys read from the Store: low \|z\| or missing cells (#409) |
| `cost.py` | what pruning costs against clumping (#406): the per-variant cM array, the pruning stages of a tier, plink2's clumping, and the ratio |

A sweep takes the selection, the map and the panel directory as inputs:

```bash
pixi run -e dev python -m prototypes.locus_leads select --stores /data/opengwasdb/stores \
    --seed 403 --output docs/benchmark-output/locus_leads/eur_sweep_selection.json
pixi run -e dev python -m prototypes.locus_leads sweep \
    --selection docs/benchmark-output/locus_leads/eur_sweep_selection.json \
    --stores-dir /data/opengwasdb/stores --genetic-map <map>.tsv.gz --panel <panel directory> \
    --hm3 /data/opengwasdb/reference/hapmap3-hg38/hm3.grch38.alid.txt \
    --work-dir <scratch directory> --output docs/benchmark-output/locus_leads/eur_sweep.json
```

The sweep also writes `eur_sweep_pairs.tsv.gz` beside its JSON: every disagreeing pair at
r² 0.01 for four configurations, with its sha256 in the JSON. Each Analysis's record is cached
under the work directory with a fingerprint of its inputs; a cache made with other inputs is
refused, not reused.

## What pruning costs (#406)

```bash
pixi run -e dev python -m prototypes.locus_leads cost \
    --genetic-map /data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/EUR.tsv.gz \
    --panel /data/opengwasdb/reference/hgdp1kgp-hg38/pgen/EUR \
    --work-dir <scratch directory for the cM arrays and plink2's files> \
    --repeats 3 --output docs/benchmark-output/locus_leads/cost.json
```

The run builds a per-variant cM array over each Store's whole Variant Table, then
for each slice times the tier's own query and the same query resolved, the
positions, the cM lookup, the sort, and both rules, and times plink2's clumping of
the same Analysis's shared set, one chromosome at a time. Each stage is timed
`--repeats` times and reported as a median with its range, a load record per
repeat, and its peak RSS with the RSS already resident when it began — the child
process's for plink2, this process's otherwise, from `/proc/<pid>/statm` rather
than `ru_maxrss`; a peak nobody observed is recorded as missing, never as zero.
The sort is reported for information only: both rules sort inside their own time,
so no total charges it. Pruning is divided by three clumping bills — plink2 alone,
plink2 with the panel match it needs, and the whole shared set. `cost.qmd`
computes every figure on the page from the committed JSON.

`cost.fast_locus_leads` is the same rule by another route: it asks a block of rows
at once whether the accepted leads already exclude them, and gives identical
leads. The run refuses to publish if the two disagree.

The map itself, its source and checksums are described in
`/data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/README.md`.

The prominence and LD-block rules (#409) run on #403's selection, sweep JSON, map, panel and
HapMap3 list, which they check against #403's records, and on the LD-block file:

```bash
pixi run -e dev python -m prototypes.locus_leads profile-sweep \
    --selection docs/benchmark-output/locus_leads/eur_sweep_selection.json \
    --sweep docs/benchmark-output/locus_leads/eur_sweep.json \
    --stores-dir /data/opengwasdb/stores --genetic-map <map>.tsv.gz --panel <panel directory> \
    --hm3 /data/opengwasdb/reference/hapmap3-hg38/hm3.grch38.alid.txt \
    --blocks /data/opengwasdb/reference/hgdp1kgp-hg38/ld_regions_hg38.tsv \
    --work-dir <scratch directory> --output docs/benchmark-output/locus_leads/eur_profile.json
```
