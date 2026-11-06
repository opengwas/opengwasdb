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
| `maps.py` | the comparison maps (HapMap II lifted, the pyrho median), and the shared-set check |
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

The sweep also writes `eur_sweep_pairs.tsv.gz` beside its JSON: every disagreeing pair at
r² 0.01 for four configurations, with its sha256 in the JSON. Each Analysis's record is cached
under the work directory with a fingerprint of its inputs; a cache made with other inputs is
refused, not reused.

The map itself, its source and checksums are described in
`/data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/README.md`.

## Other genetic maps (#404)

The same sweep runs on any map in the layout above. Three are compared:

```bash
# the median of the five pyrho EUR maps, the alternative to their mean's two spikes
pixi run -e dev python -m prototypes.locus_leads build-median-map \
    --archive /data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/original/hg38_maps.tar.gz \
    --output /data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/EUR-median.tsv.gz

# HapMap II lifted to GRCh38, as Eagle ships it, normalised and checked
pixi run -e dev python -m prototypes.locus_leads build-hapmap2-map \
    --original /data/opengwasdb/reference/genetic-map/eagle-hapmap2-hg38/original/genetic_map_hg38_withX.txt.gz \
    --output /data/opengwasdb/reference/genetic-map/eagle-hapmap2-hg38/HapMap2.tsv.gz

# every map must leave the same shared set, row for row, before its counts mean anything
pixi run -e dev python -m prototypes.locus_leads map-shared-sets \
    --selection docs/benchmark-output/locus_leads/eur_sweep_selection.json \
    --stores-dir /data/opengwasdb/stores \
    --panel /data/opengwasdb/reference/hgdp1kgp-hg38/pgen/EUR \
    --genetic-map pyrho-mean=/data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/EUR.tsv.gz \
    --genetic-map pyrho-median=/data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/EUR-median.tsv.gz \
    --genetic-map hapmap2=/data/opengwasdb/reference/genetic-map/eagle-hapmap2-hg38/HapMap2.tsv.gz \
    --reference pyrho-mean \
    --output docs/benchmark-output/locus_leads/eur_maps_shared_set.json
```

`build-median-map` and `build-hapmap2-map` write a sidecar beside the map naming
what it came from — the archive's md5, or the download's url and sha256 — and
`load_map` checks it on every run. A map built from a download rather than the
pyrho archive records the download's sha256 in place of the archive's md5.

A map in another layout is refused, not read by column number; a map missing a
chromosome is refused, not left to give NaN cM; and cM that falls with position
costs the points that break the running maximum, the build refusing a chromosome
that would lose more than 0.1% of its points to it. `HapMap2.json` and
`EUR-median.json` record what each map lost, and the READMEs beside them quote
the sources, retrieval dates, checksums and licences.

`compare-maps` puts each sweep's count criteria side by side, and refuses to
write when the maps' shared sets differ.
