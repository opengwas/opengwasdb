- **A 0.3.0 release is queried without reading `variants.tsv.gz` (issue #469, ADR 0105
  §7).** `variants_table()` and
  `resolve(..., include_variant_info=True)` answer from the sidecars: the ALID sidecar
  names each row, its ALID gives the chromosome, position and alleles, and the rsID index,
  inverted, gives the rsID. The variant count is the ALID sidecar's. No query loads pysam,
  through the Python API or `opengwasdb query-*`: the CLI imports Reference Completion,
  which does, only inside the `complete-*` commands.
  The answers equal 0.2.x's column by column: a row with no rsID is `None` in
  `variants_table()` and `.` in a resolved row, and a cell naming several (`rs1;rs2`) is
  that cell whole. Naming a few rows scans the rsID index (0.04 s for one row on
  OGS-00004's 23.7 M); once the scans would cost more than one inversion (1.0 s, 95 MB),
  it inverts, once, and keeps it. 0.2.x is unchanged.
- **`variant_offsets.npy` is not written from 0.3.0.** A 0.3.0 release carrying it fails
  `envelope.closed`. The Dense builder's `provenance.dense.variant_axis.row_offsets` no
  longer names it, and the `index.sqlite` `dense` blob that also named it is gone from
  builds and rewrites alike (#471). `variants.tsv.gz` and its `.tbi` still ship, written with htslib,
  as a convenience for tabix, bedtools and R. Spec §4 gives the command that rebuilds a lost
  `.tbi`: `tabix -s1 -b2 -e2 -c'#' variants.tsv.gz`.
- **`overview.html`'s file guide says so.** `variants.tsv.gz` is a file to open directly,
  which no 0.3.0 query reads, and every sidecar and `regions.tsv` is described rather than
  listed as "Internal store file". `--variant-info`'s help names the rsID index it reads.
- **A 0.3.0 release without its rsID index is refused** (`MissingRsidIndex`), rather than read
  as naming no rsID. A 0.2 release built before #109 still opens without one.
- **The variant-axis writer refuses an rsID wider than 24 bytes** (`UnindexableRsidError`).
  0.2.x left such a name out of the rsID index and kept it in the table. A 0.3.0 reader names
  rows from the index, so the name would vanish from every answer. None of the six
  releases measured holds one (OGS-00003, -00004, -00007, -00009, -00016, -00025: 0 cells
  over 24 bytes). The rebuild reads sources, though, and a GWAS-VCF `ID` cell is kept
  whole: one naming several rsIDs past 24 bytes (three of nine digits already are) is
  refused at build.
- **`ogdb rewrite-release` gains a `variant_offsets` step**: the copy
  leaves the file out, and the step refuses unless the ALID and position sidecars count the
  rows the source's offsets did.
- **New validation rule `variant_table.rsid_rows`** (0.3.0). It checks that the inverted rsID
  index names every row by the table's rsID. On 0.3.0, `variant_table.rows` counts against
  the ALID sidecar, and `variant_table.representatives` checks the tabix index only.
- **ogdb-api breaks on a 0.3.0 release.** It reads `variant_offsets.npy` and seeks into the
  Variant Table through the axis's private attributes. Its owner moves its pin.
- opengwasdb-stores' release manifests and walkthrough name the envelope's files, so they
  change when opengwasdb-stores moves its pin to this version; no change there is open yet.
