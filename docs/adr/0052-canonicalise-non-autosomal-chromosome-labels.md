# Canonicalise non-autosomal chromosome labels

## Context

ALID begins with a chromosome label, so two spellings of one physical
chromosome create two variant identities. `normalise_chromosome` already removed
a `chr` prefix and normalised letter case, but it left PLINK-style numeric
labels distinct: the same variant could therefore become both
`X:154532223:A:G` and `23:154532223:A:G`. Those rows never collided in a Store
Variant Table and a cross-Analysis query silently returned only one namespace.

This is common in the GWAS-Catalog EUR hybrid source pool measured for issue
#216. Of a stratified 384-file sample, 69 sources carried X rows: 36 used `23`
and 33 used `X`. Y and mitochondrial rows also occurred under numeric and
letter spellings.

## Decision

`normalise_chromosome` owns a closed alias table for the non-autosomal physical
chromosomes:

| source spellings after optional `chr` removal and case folding | canonical label |
|---|---|
| `23`, `X` | `X` |
| `24`, `Y` | `Y` |
| `25`, `26`, `M`, `MT` | `MT` |

`25` is included because the measured GWAS-Catalog sources in scope use it for
mitochondrial rows, alongside `M` and `MT`; `26` is the PLINK mitochondrial
code. A pipeline carrying PLINK's `25` = XY/PAR convention must resolve those
rows to assembly X/Y coordinates before this normalisation seam. OpenGWASDB has
no XY/PAR chromosome identity to preserve.

Autosomes `1` through `22` remain unchanged. Other non-empty, non-numeric
contig labels also remain unchanged: they are outside this alias table rather
than guessed into a physical chromosome. Numeric labels outside the enumerated
`1` through `26` set, and a missing label including bare `chr`, are rejected.
Every reader path, including projected hot paths, applies the same mapping
before constructing an ALID.

Pseudoautosomal regions are out of scope. The package has no separate `PAR`,
`PAR1`, or `PAR2` identity model; an X-coordinate remains on canonical `X` and
this decision does not remap coordinates between assemblies or chromosome
names.

## Consequences

- Sources spelling the same X, Y, or mitochondrial variant differently now
  produce the same ALID and collide into one Store Variant Table row.
- This changes variant identity. A Store built before this decision that
  contains numeric `23`–`26` or `M` ALIDs cannot be safely reference-completed
  or joined on those non-autosomes against a post-change Store. Affected Stores
  must be rebuilt.
- The Store format version does not change. This is a source-to-ALID identity
  correction, not a new interpretation of stored bytes: `X`, `Y`, and `MT`
  were already valid ALID chromosome labels, so an old `0.1.0` reader reads a
  newly built Store correctly. This differs from ADR 0038's major-change cases,
  in which an old reader misdecodes a new Store. The reverse direction is
  deliberately not supported: affected old Stores with numeric or `M` ALIDs
  must be rebuilt because a new reader cannot safely infer which convention
  produced them.
- Alternate contigs are not rejected or rewritten. Inventing a general contig
  naming scheme without an assembly-specific authority would risk conflating
  distinct sequences.

## Rejected

- **Canonical numeric labels.** `23`/`24`/`25` are PLINK conventions rather
  than assembly chromosome names, and `26` is also used for mitochondrial DNA.
  `X`, `Y`, and `MT` state the physical chromosome explicitly.
- **Fixing each registry acquisition pipeline.** All supported source readers
  feed this package's normaliser. Duplicating the mapping upstream would leave
  ALID identity dependent on which acquisition path happened to run.
- **Treating pseudoautosomal regions as another alias.** PAR identity needs an
  explicit coordinate and assembly policy; a spelling table cannot supply one.
