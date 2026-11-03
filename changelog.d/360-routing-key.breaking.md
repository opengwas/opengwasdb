- **`opengwasdb.variants.routing.RoutingKey` is removed (issue #360, ADR
  0075).** Its two modes, the source's own spelling for Dense and upper-cased
  alleles for Hybrid, now spell every key alike, so there is nothing left to
  choose: `route_variants` and `RouteTable.build` no longer take `key=`. The
  spelling is `opengwasdb.variants.routing.keys.routing_key`, its batch form
  `encode_routing_keys`, and a site's own ALID `source_alid`; the rule behind
  all three is `opengwasdb.variants.normalise.identity_allele`, and an ALID is
  formed from raw alleles by `canonical_alid`. `route_variants` takes `label=`,
  what a harvest's refusals and warnings call the build.
