# Locus Leads against plink2 clumping (throwaway prototype)

Epic #62 proposes one **Locus Lead** per locus, chosen by genetic distance: take
an Analysis's Top-Hit Query rows in descending `abs_z` (ties by ascending Store
Variant Index), and accept a row unless an accepted lead lies within W bp and
D cM. This prototype measures that rule against plink2 LD clumping on the
HGDP+1kGP genotype panels before anything is built into a Store. It is not part
of the package and will be deleted once the epic's decisions are made.

Run from the repository root:

```bash
# once: average the pyrho EUR population maps into one file (--ancestry AFR, EAS
# or AMR writes that ancestry's map beside EUR's)
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
| `eur_map.py` | a map averaged from the pyrho population maps, EUR's by default |
| `ancestries.py` | which pyrho populations each ancestry's map is averaged from |
| `panel.py` | the genotype panel, and matching Store rows to it |
| `clump.py` | -log10 p, tie separation, plink2 clumping |
| `compare.py` | spurious splits and lost signals |
| `run.py` | one run, end to end; clumping is independent of D and W, so a sweep reuses it |
| `select.py` | the seeded, stratified draw of Analyses for one ancestry, and its record |
| `sweep.py` | every (D, W) and the bp-only baseline over the drawn Analyses; pooled rates and the working criteria |
| `pairs.py` | each disagreeing pair's r², distance band and spike, chance LD between chromosomes, and the pair file |
| `counts.py` | the count criteria C1-C3: rank, density against HapMap3, cM against bp-only |
| `ld.py` | r² between panel variants, from dosages plink2 exports |

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

Another ancestry is the same two commands with `--ancestry AFR`, `EAS` or `AMR` on
`select`, that ancestry's map and panel on `sweep`, and `--spikes none` because the
spike sets #402 found are the EUR map's. The |z| quotas and the target floor at
what the pool holds; the record keeps how many Analyses each stratum holds.

The sweep also writes `eur_sweep_pairs.tsv.gz` beside its JSON: every disagreeing pair at
r² 0.01 for four configurations, with its sha256 in the JSON. Each Analysis's record is cached
under the work directory with a fingerprint of its inputs; a cache made with other inputs is
refused, not reused.

The map itself, its source and checksums are described in
`/data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/README.md`.
