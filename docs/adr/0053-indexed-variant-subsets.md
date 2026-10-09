# Indexed Variant Subsets are optional full-statistic Dense query indexes

Issue #262 asks for extraction of the approximately 1.2 million HapMap3 variants
from one Analysis in under one second; issue #263 specifies the implementation. On OGS-00009 a membership index alone does
not do that: those variants touch 9,822 of the Dense Z plane's 9,848 row chunks,
so selecting their Store-local Variant Indices still takes a warm median of 15.37
seconds. This decision adds an optional Analysis-major derived index, accepts its
bounded storage cost, and keeps the primary statistic planes authoritative.

## Decision

An **Indexed Variant Subset** is a named, optional, rebuildable Dense-store query
index over a caller-supplied set of canonical ALIDs. It contains the subset for
**every Analysis**; there is no partial-Analysis mode. It is stored alongside the
other optional derived Zarr artifacts:

```text
data.zarr/
  rho/
  top_hits/
  indexed_subsets/
    <name>/
```

Each subset stores Store Variant Indices and enough Analysis-major statistic
planes and encoding side tables to reproduce the release's ordinary association
result over those variants. That means Z and SE always, EAF when the release has
it, and Association Status for a Reference-Completed release. Beta and p-value
remain derived from Z and SE, and Analysis-level sample size remains in
`analyses.tsv`; none is duplicated into the index.

There is **no Z-only profile**. On the measured OGS-00009 release, the complete
HapMap3 index is 5.211 GB (15.0% of the 34.681 GB release), versus 3.676 GB
(10.6%) for Z alone: full statistics cost 1.42x, not 2x. The extra state and
query contract of two profiles are not justified by saving 4.4% of the source
release, especially when SE and EAF make the same index useful for instrument
extraction and beta reconstruction. Residual-coded SE is also unreadable without
the EAF it was coded against, so SE and its EAF dependencies are one indivisible
bundle.

The public build and query interface follows the existing flat CLI:

```text
opengwasdb build-indexed-subset STORE SUBSET_NAME \
  --variant-list hm3.alid.txt --reference-assembly GRCh38

opengwasdb query-analysis STORE ANALYSIS_ID --indexed-subset SUBSET_NAME
```

The variant-list assembly is mandatory because cross-Store Variant Identity is
Reference Assembly plus ALID. Generation rejects a mismatched assembly,
non-canonical or duplicate ALIDs, an invalid subset name, and a list with no
Store matches. ALIDs absent from the Store Variant Table are permitted but
counted and recorded; they are never silently treated as present. The index
stores its input checksum, assembly, requested/resolved counts, encoding and
build provenance.

A query through an Indexed Variant Subset has the same result fields,
orientation, missing-cell filtering and genomic ordering as filtering the
ordinary `analysis()` result to the same Store Variant Indices. Requesting an
unknown, incomplete, stale or invalid subset fails loudly; it never silently
falls back to the primary 15-second path. A caller that does not request a
subset sees unchanged query behaviour.

Indexed Variant Subsets are **non-authoritative derived artifacts**, like the
Top-Hit Index and Rho Matrix. They may be added, atomically replaced or removed
in place without minting a Store Release because they change no association or
Analytical Metadata and an unaware reader still reads the primary planes
correctly. Generation stages the complete named group and publishes it under a
release-local lock. Validation checks its identity, shapes, Variant Indices,
encoding dependencies and decoded values against the primary planes. Deleting
the group leaves a valid release whose ordinary queries are unchanged.

They do not move `format_version`. This narrows ADRs 0038 and 0041: their
“optional array, index, or sidecar” compatibility example applies to optional
Store-format content a reader may need to interpret authoritative data, not to
a self-describing acceleration artifact that can be deleted without changing
any existing query answer. The package and specification learn the optional
namespace, while the Store Release's authoritative representation remains the
same format.

The initial index is Dense-only. Ragged already stores each Analysis as a direct
CSR slice, while a Hybrid index would have to specify how its Dense and Ragged
Overflow Components are unified; neither should be implied by a Dense
performance result.

## Considered options

- **Membership index only.** Rejected: its approximately 11 MB footprint is
  attractive, but HapMap3 intersects 99.736% of OGS-00009's row chunks and the
  measured query remains 15.37 seconds, missing the target by over an order of
  magnitude.
- **Z-only and full profiles.** Rejected initially: the full profile adds 1.535
  GB to the measured HapMap3 artifact and removes an entire format, validation
  and query-state branch. A later measured need may add a new explicit profile;
  arbitrary per-plane switches are not admitted.
- **Per-Analysis indexes.** Rejected: they add partial coverage, cold-query and
  discovery states for little benefit. An Indexed Variant Subset covers every
  Analysis or does not exist.
- **External projection artifact.** Rejected: callers would have to locate and
  match a second path to a release. The established Rho and Top-Hit pattern puts
  optional derived artifacts under `data.zarr` and validates them with the
  release they accelerate.
- **A new Store Release.** Rejected: the index changes no authoritative
  association data or metadata and is entirely rebuildable from the release and
  variant list.
- **“Projection” as the public term.** Rejected: it describes the array/database
  implementation but not the user concept. “Indexed Variant Subset” states both
  the selected domain object and its acceleration purpose.

## Consequences

- OGS-00009 users may choose no extra storage or one 5.211 GB HapMap3 index; no
  Store Release pays the cost by default.
- The prototype's Z path returns 1,178,549 present associations for
  `ukb-b-17805` in 11.9 ms median and is byte-equivalent after decoding,
  including exact overflow values. Full-result latency must be measured during
  implementation rather than inferred from that Z-only timing.
- The store format gains a compatible optional index namespace and validation
  rules. The specification, overview rendering, CLI documentation and sibling
  repository walkthrough must describe it in the implementation change.
- The existing broad immutability wording must distinguish authoritative Store
  contents from rebuildable derived indexes, making the already-shipped
  Top-Hit/Rho in-place precedent explicit rather than leaving it exceptional by
  accident.
