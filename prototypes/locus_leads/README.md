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

The map itself, its source and checksums are described in
`/data/opengwasdb/reference/genetic-map/pyrho-1kg-hg38/README.md`.
