# LD is served by a separate service, ogld-api, from local disk; the LD Stores will also be public as static Zarr

Formerly ogld ADR 0008.

> **The Proxy Index is gone** ([ADR 0097](./0097-partner-rows-replace-the-proxy-index.md), opengwas/ogld#72).
> Where this record speaks of it in the present tense — as a component a store holds, a stream a
> build writes or a read a query makes — read it in the past tense: the Partner Rows hold every
> entry it held, and `clump` and `proxies` read them. Everything else here stands.

Issue [#9](https://github.com/opengwas/ogld/issues/9), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
The reads are in `docs/spec/ld-store.md` §10 and their compatibility in §13. The measurements
are in `docs/benchmarks/ld/latency.md` (ogld#8), `docs/benchmarks/ld/fast-reader.md` (ogld#32),
`docs/benchmarks/ld/far-pairs.md` (ogld#31) and `docs/benchmarks/ld/serving.md` (this ticket), each
rendered from records under `docs/benchmarks/ld/results/`. The terms are in `CONTEXT.md`.

**Who decided.** The owner, on 13 November 2026, in the orchestrated run, recorded on ogld#9,
from this ADR's proposal on `orch/1/9-serving` and the numbers in `docs/benchmarks/ld/serving.md`;
this ticket drafted the proposal, and the owner's rulings made it a decision. Their three
rulings are the three decisions below. The proposal had recommended ogdb-api's job workers, with
LD jobs in worker processes of their own; the owner chose a separate service, and that option is
kept below with what it would have cost. They also answered the question this
record had left open: ogdb-api's live server runs on this IEU node, app-dc3-ogws-p0.

## The questions

1. **The serving route.** (a) ogdb-api runs `opengwasdb.ld` as a library in its job workers, which
   already run queries as jobs; or (b) a separate service in `opengwas/ogld-api`.
2. **Static Zarr.** Whether the LD Stores are also served as static Zarr that clients read
   themselves with `opengwasdb.ld.open("https://…")`.
3. **Where the five HGDP+1kGP LD Stores are published.** This moved here from ogld#7 (owner,
   11 November 2026).

## Context

**The stores.** The five stores of ogld#7 hold 533.4 GB in 58,123 files: Near Bands 486.1 GB,
Proxy Indexes 45.2 GB, Far Pairs 1.46 GB (ogld#31) and variant tables 0.56 GB, over 47.2 M
variants. AFR alone is 241.2 GB in 17,009 files. Each is immutable once validated, format
`0.1.0` (§13), layout C (ADR 0092).

**ogdb-api as it is** (opengwas/ogdb-api at `3f70e0e`, read on 13 November 2026). ogld-api takes
its job contract from here where it fits (Consequences):

- **Where it runs.** A Compose deployment, an `api` container and Redis, with `/data/opengwasdb`
  mounted read-only. Its owner's instance, port 6433, runs on the IEU node that holds
  `/data/opengwasdb` and the staged LD Stores, and reads the engine's Stores from local disk; its
  agents' VM runs on the same node and mounts `/data` over SSHFS. On 13 November that instance
  served 13 Stores with `backend_max_running` 180, in one uvicorn process of 316 MB resident.
  Its live server runs on the same node (owner, 13 November 2026, answering this ticket).
- **How a query runs.** Every data query is a job keyed by (user, canonical query hash) in a
  Redis queue (its opengwas/ogld#11). Each API process runs **one** worker coroutine, which claims one job at
  a time and runs its store query on a bounded thread pool in the same process
  (`ogdb_api/operations.py`); more throughput means more API processes on one Redis, whose
  admission is shared (its opengwas/ogld#18). Its stress run measured 0.2 jobs/s for one process, and "latency
  grows with queue depth, not with memory" (DEVLOG, its opengwas/ogld#10).
- **Limits a job meets.** A store query that runs past `OGDB_STORE_TIMEOUT_SECONDS` (30 s) fails
  terminally; the query cannot be interrupted, so its thread and release slot stay held until it
  returns (`ogdb_api/planner.py`, `BackendLimiter`).
- **How a result is returned.** The client polls `GET /api/jobs/{id}` at the interval the API
  recommends, then reads the result: inline up to 10,000 rows, else immutable NDJSON pages, at
  most 100,000 rows and 50 MB a job, held in a results cache of 1 GiB until every page is read
  plus 300 s (its opengwas/ogld#16).
- **Memory.** Its specification says memory, not CPU, is the binding constraint: OGS-00011's
  reader holds about 24 GiB, and on a 32 GiB VM the whole deployment was OOM-killed at 30.9 GiB.
- **LD today.** `/api/ld/clump`, `/api/ld/matrix` and `/api/ld/reflookup`, OpenGWAS's own LD
  paths, are registered and answer `501 Not yet implemented` until "an explicitly supported LD
  panel and algorithm" exists (its SPECIFICATION §5). Its ieugwasr and ieugwaspy compatibility
  epic (its opengwas/ogld#35) keeps those clients' paths in ogdb-api.
- **Versioning.** It pins `opengwasdb` to one Git revision and upgrades it deliberately, by a
  ten-step procedure in its README.
- **In front of it.** opengwas/opengwas-gateway authenticates users (JWT) and forwards
  `/{namespace}/{path}` to one backend per namespace, `ogdb` the only one so far.

**The engine.** Engine issue opengwas/opengwasdb#419 makes the engine's own Top-Hit Query clump
on the fly with `ogld.clump`, against an LD Store the caller names, when the configuration is not
one of the Store's stored Clumps (#418). ogdb-api serves that query, so under either route ogdb-api
imports `opengwasdb.ld` and reads LD Stores for top hits. The engine's roadmap holds that a store "should
support laptop, HPC, and cloud/API usage" and that "the service layer should route and authorise
queries, not become the only place the data can be used" (`docs/opengwas-roadmap.md`, principle
2), and that workers may "mount stores from local disk, object storage, or HPC filesystems"
(Horizon 4). It has no ADR on serving over HTTP or on where Stores are hosted: the engine's ADR 0018 puts
remote querying and store discovery in "a separate catalogue or deployment layer", and its Stores
live under `/data/opengwasdb/stores/OGS-*` on the same node, listed by opengwasdb-stores' Analysis
Catalog (opengwasdb-stores ADR 0039), which records no path.

**What the reads cost**, on the published EUR store, from local disk and over HTTP with a wait
added to every request. "Now" is the reader with ogld#31's Far Pairs and ogld#32's changes, timed
by this ticket (`docs/benchmarks/ld/serving.md`); ogld#31 changed no clump or proxy read.

| read | local disk | 10 ms a request | 50 ms | 100 ms |
|---|---|---|---|---|
| clump #403's 37 Analyses, r² 0.01, cold, now | 442 s (7.4 min); 48.6 GB read | — | — | — |
| the same, ogld#8's reader | 1,043 s (17.4 min); 75 GB read | about 71 min, estimated | about 3.8 h, estimated | — |
| clump ogld#8's four Analyses, r² 0.01, cold (ogld#32) | 52 s | 102 s | 224 s | — |
| matrix of 500 chr22 variants, cold, now | 1.2 s | 2.9 s | 6.8 s | 11.8 s |
| matrix of 500 variants genome-wide, cold, now | 8.3 s | 15.4 s | 36.7 s | 61.2 s |
| `proxies` for 1 variant, store open, now | 29 ms | 108 ms | 346 ms | 655 ms |
| `proxies` for 100 variants, store open, now | 1.1 s | 2.8 s | 7.0 s | 12.2 s |

The HTTP rows are loopback with a wait injected per request, not a real WAN: nothing limits
bandwidth, so they leave out what the bytes cost on a real link (48.6 GB is 6.5 minutes at
1 Gbit/s).

## Decision

### 1. Route (b): LD queries are served by a separate service, `opengwas/ogld-api`

ogld-api runs `opengwasdb.ld` as a library against the LD Stores on local disk and answers LD queries
(`clump`, `matrix`, `region`, `proxies`, and a presence lookup for ieugwasr's `reflookup`). Its
epic, drafted in ogld#9's report, is filed in `opengwas/ogld-api` as
[opengwas/ogld-api#1](https://github.com/opengwas/ogld-api/issues/1) and placed in the LD
workstream, as ogld#9's acceptance criterion asks. It runs on the same
node as ogdb-api and the stores, so it reads them from local disk: an Analysis's clump takes
6.3 s at the median and 69 s at most, and a matrix of 500 chr22 variants 1.2 s, before the job's
round trips. LD is deployed, scaled, versioned and restarted apart from ogdb-api, and an LD
failure, an OOM included, cannot take association queries with it.

### 2. Static Zarr: yes, public

Anyone may open an HGDP+1kGP LD Store directly with `opengwasdb.ld.open("https://…")` (owner). The
store directories are published as they are, where a static host serves them with byte ranges,
and the reader reads them as it reads a local store (§10: "a static host that honours byte
ranges is enough"). No code is needed: the reader, its refusals and its answers are the same over
HTTP, and ogld#8, opengwas/ogld#31, opengwas/ogld#32 and this ticket measured them. It is the route for clients near the
data and for reproducing a result against a named store, not the interactive route for a remote
user: a 500-variant genome-wide matrix is 8.3 s locally, 36.7 s at 50 ms a request and 61.2 s at
100 ms, a third of it loading the chromosomes' arrays at those latencies, and a clump reads Near
Band chunks by the gigabyte, 48.6 GB for #403's 37 Analyses.

**Until a public host is chosen (decision 3), no store is served over HTTP.** This ruling decides
that the stores will be public as static Zarr; the host is decided later.

### 3. Location: local disk only, for now

The stores are served from `/data` on this IEU node, app-dc3-ogws-p0, where ogdb-api's live
server runs and ogld-api will. Public hosting, and so the static Zarr of decision 2, is decided
later, as a follow-up; the options and their costs are under "Considered options", kept as the
record for that decision.

**The served copy's path.** The stores are being rebuilt into
`/data/opengwasdb/work/ogld/stores-final/` and swapped into `/data/opengwasdb/work/ogld/stores/`
by epic opengwas/ogld#1's final lane. That path is a work area: builds, benchmarks and lanes write there, and
a swap replaces a directory a running service may be reading. This record proposes a stable path
for the served copy, **`/data/opengwasdb/ld-stores/<store name>/`**, for example
`/data/opengwasdb/ld-stores/hgdp1kgp-v2-EUR-k10.store/`: beside the engine's
`/data/opengwasdb/stores/`, outside the Analysis Catalog's root, each store written once,
validated, made read-only and never changed, and a rebuilt store published under a new name.
The owner ruled `/data` on this node, not the directory, so the path is open (below) until they
confirm it. Until then, the stores are at `/data/opengwasdb/work/ogld/stores/` after the swap.

## Considered options

### The route

| | (a) ogdb-api's job workers | (b) a separate ogld-api service |
|---|---|---|
| where it reads | local disk, same node | local disk, if deployed on the same node |
| latency | local: an Analysis's clump 6.3 s at the median, 69 s at most; 500 chr22 variants 1.2 s; plus the job's polling | the same |
| job contract | ogdb-api's, as it is | to build: queue, keys, polling, retention, controls, metrics, admission |
| result | ogdb-api's pages; a matrix as one row per variant, its r beside it | its own |
| memory | one LD worker process: 1.5 GiB with the five stores' chromosomes loaded, 21.5 GiB with their caches full at the default 4 GiB; `cache_bytes` sets it | the same |
| failure isolation | per process, with LD workers of their own; none without them | per service |
| versioning | ogdb-api pins ogld beside opengwasdb; a reader change redeploys the API image | ogld-api pins ogld alone; deploys on its own |
| top-hit clumps (opengwasdb#419) | the same process and pin | still ogld inside ogdb-api: two pins of ogld, both of which must read every served store |
| gateway | `ogdb`'s paths, unchanged | a second namespace, or ogdb-api forwarding `/api/ld/*` |
| to build | LD operations, an LD worker role, the readers kept open, a timeout per operation | a service, its deployment, and all of ogdb-api's job system again |

- **(b), ogld-api.** Chosen. It isolates LD in a service of its own, deployed and versioned
  apart from ogdb-api. It costs a job system of its own, built from ogdb-api's contract, and a
  second service to run, and it does not remove ogld from ogdb-api (opengwasdb#419).
- **(a) as ogdb-api is, with LD jobs in its one worker.** Rejected. Under ogdb-api as it is, an
  LD job shares a process with association jobs, and that costs three things:
  - _the queue_: one worker a process runs one job at a time, so a clump, 6.3 s at the median
    and up to 69 s, holds up every association job behind it;
  - _the timeout_: 4 of #403's 37 Analyses clump past ogdb-api's 30 s at r² 0.01, the largest
    (260,821 hits) in 69 s, and 1 at r² 0.001;
  - _memory_: a process with the five stores open holds 1.45 GiB of chromosome arrays once it
    has met every chromosome, and 21.5 GiB resident once their 4 GiB caches fill (peak
    22.1 GiB), where ogdb-api's 32 GiB VM was OOM-killed at 30.9 GiB without them.
- **(a) with LD jobs in worker processes of their own.** This ticket's proposal; not chosen. It
  would have reused ogdb-api's job system whole and removed the three costs by process
  isolation, but LD would have been released with ogdb-api's image and its ogld pin. Two other
  costs of sharing a process were measured. While ogld reads on a worker thread, an asyncio loop
  on the main thread wakes 2–16 ms late at the 99th percentile, and once in each of two settings
  more than 100 ms late (at most 206 ms), against 0.2–0.3 ms idle. And under the zarr codec
  pipeline the engine sets process-wide when imported (opengwasdb ADR 0056), every query took
  longer, round against round: a clump 17% and 49%, a matrix 19% and 45%, a 3 Mb region 12% and
  14%, with the same answers, at a node load of 55–94.
- **Synchronous LD endpoints, no job.** Rejected by this record, not by the owner's ruling,
  which chose the route: it infers from ogdb-api's rule that every data query is a job and that
  direct routes go (its owner, 6 November 2026, its opengwas/ogld#11 and opengwas/ogld#14), with ieugwasr's and ieugwaspy's
  compatibility paths the one exception that waits server-side, on a job. A clump can run for
  minutes, which a direct route would hold open.

### Static Zarr

- **Yes, public.** Chosen. No code; the store's own refusals and NaN travel with it. Remote
  clients pay round trips and bytes, and the host pays egress unless it is free.
- **No: the API only.** It keeps every read behind authentication and in the usage logs, and
  no egress is paid, but a store can then be used only through the service, against the
  engine's principle 2, and a cluster user moves every result through polling.
- **Yes, behind authentication** (signed URLs, or a gateway in front of the static host). It
  keeps the usage record, at the cost of a token every client must obtain. Possible later for
  a store whose terms need it; the HGDP+1kGP stores do not (their genotypes are public under
  gnomAD's terms, ADR 0088 decision 7; ADR 0087 decision 5 allows LD Stores to be published).

### Where

The owner chose local disk only, for now (decision 3). This table is kept as the record for the
later choice of a public host. List prices are for 533 GB, each read on 13 November 2026 from
the source under the table; where the provider's own page would not load, the source is a
third-party summary and says so. "A full clump" is the 48.6 GB, in 86,286 requests, that #403's
37 Analyses read at r² 0.01 (`docs/benchmarks/ld/serving.md`).

| place | storage a month | egress | a full clump read remotely | access control | read locally by a service on the node |
|---|---|---|---|---|---|
| the IEU node, a static web server | none new (533 GB of 18 TB free) | the University's terms were not found | no charge found; bounded by the node's link | the web server's | yes |
| Cloudflare R2 | about $8 | free | about $0.03 of read requests | public bucket, or tokens | no |
| OCI Object Storage | about $13.60 | 10 TB a month free, then about $0.0085/GB | free within 10 TB | pre-authenticated requests, policies | no |
| AWS S3 Standard | $12.26 | $0.09/GB after 100 GB a month | about $4.40 | bucket policy, signed URLs | no |
| Backblaze B2 | about $3.70 | free to 3× stored a month (1.6 TB), then $0.01/GB | free within 1.6 TB | public bucket, or keys | no |
| a CDN in front | the origin's | CloudFront about $0.085/GB in Europe after 1 TB free; Cloudflare's CDN terms restrict large non-HTML files not hosted on R2 | the origin's, less cache hits | the CDN's | no |
| Zenodo | free | free | — | public | no |

Sources, each read on 13 November 2026:

- **Cloudflare R2:** the provider's page, <https://developers.cloudflare.com/r2/pricing/>:
  $0.015 per GB-month, Class B reads $0.36 a million, egress free, 10 GB-month free.
- **OCI Object Storage:** third-party summaries, because Oracle's pricing pages answered 403
  (<https://www.activepieces.com/blog/oracle-cloud-infrastructure-pricing-2026-price-guide>, and
  Oracle's <https://www.oracle.com/cloud/networking/virtual-cloud-network/pricing/> as search
  results quote it): about $0.0255 per GB-month, 10 TB of egress a month free and then about
  $0.0085/GB. One summary says the allowance changed in February 2026; unconfirmed.
- **AWS S3 Standard:** the provider's page, <https://aws.amazon.com/s3/pricing/>, for GET requests
  ($0.0004 per 1,000) and the 100 GB of free egress a month; third-party summaries for storage
  ($0.023 per GB-month for the first 50 TB) and egress ($0.09/GB for the first 10 TB), as the
  provider's page did not show them
  (<https://www.cloudforecast.io/blog/aws-s3-pricing-and-optimization-guide>). The Open Data
  Sponsorship Program: <https://aws.amazon.com/opendata/open-data-sponsorship-program/>.
- **Backblaze B2:** the provider's page, <https://www.backblaze.com/cloud-storage/pricing>, as
  search results quote it: $6.95 per TB-month, egress free to 3× the average stored, then $0.01/GB.
- **A CDN:** CloudFront from search results quoting <https://aws.amazon.com/cloudfront/pricing/>
  and third-party summaries ($0.085/GB in Europe after 1 TB a month free); Cloudflare's CDN terms
  from <https://blog.cloudflare.com/updated-tos/>.
- **Zenodo:** <https://help.zenodo.org/docs/deposit/manage-files/quota-increase>: 50 GB and 100
  files a record, 200 GB on request.

- **The IEU node.** Free, and where ogdb-api and ogld-api run; chosen as the served copy. As a
  public host it would make a compute node a public web host, which needs the University's
  network and security approval, and its durability is the node's backups.
- **Cloudflare R2.** This ticket proposed it for the public copy: no egress fee, about $8 a month.
  OCI was the alternative if OpenGWAS's existing account made one vendor simpler.
- **AWS S3.** Egress makes every remote full clump cost about $4.40. Its Open Data Sponsorship
  Program can cover a public dataset's costs on application; worth asking if S3 is wanted.
- **B2.** The cheapest storage, but egress past 1.6 TB a month is charged, and 33 full clumps
  a month reach it.
- **A CDN.** LD reads are a long tail of chunks, so an edge cache saves little beyond the
  metadata and `indptr` every open reads.
- **Zenodo.** A record holds at most 50 GB and 100 files, 200 GB on request, and has no
  directories, so it cannot hold a Zarr hierarchy; AFR alone is 241 GB in 17,009 files.
  Rejected for serving; a DOI for a store's provenance is a separate question.

## Consequences

- **ogld-api builds a job system of its own, from ogdb-api's contract.** Every query is a job
  keyed by (user, canonical query hash); submission returns a job and a polling interval, and
  results are kept until read. ogdb-api's queue, keys, polling, retention, queue controls,
  metrics and Redis-shared admission are the model, reused where they fit. Whether they are
  copied or shared as a package is the epic's question.
- **A Refusal is passed to the caller, never worked around.** An `ogld.Refusal` (§9) reaches the
  caller with its reason and the store's t_far, `near_bp` and `far_bp`, as a request error that
  fails the same way every time. It is never retried at a smaller window or a higher threshold,
  and never answered approximately. A result carries its Absent ALIDs, and a matrix keeps its
  NaN: an unstored pair is never a 0 on the wire, unless the caller asked for a `fill`.
- **Memory is sized for five stores.** One process with the five stores open holds 1.5 GiB once
  it has met every chromosome, plus each store's `cache_bytes` once it fills, plus the reads in
  progress: 21.5 GiB resident at the default 4 GiB a store (peak 22.1 GiB).
- **A clump can run for minutes, and cannot be interrupted.** The worst of #403's 37 Analyses
  took 69 s locally on a node at load 5–56, where local timings vary by up to 45% (ogld#8); a
  timeout must sit well above that, and a query past it keeps its thread until it returns.
- **ogld is pinned in two places.** ogld-api pins ogld; ogdb-api reaches ogld through the engine
  for top-hit clumps (opengwasdb#419). Both read the same stores, and a reader refuses a
  `format_version` or encoding it does not implement (§13), so a store in a new format waits
  until both pins implement it, and is published under a new name beside the old.
- **ogld inside the engine reads under the engine's zarr settings.** Route (b) keeps ogld-api's
  reads under zarr's defaults, but the engine's on-the-fly top-hit clumps (opengwasdb#419) run
  ogld in an engine process, under `FusedCodecPipeline` with one worker, where every query took
  12–49% longer here. That is the engine's to weigh.
- **The gateway gains a backend.** opengwas-gateway forwards `/{namespace}/{path}` to one backend
  per namespace; ogld-api is a second. Where ieugwasr's `/ld/*` paths go is the epic's question.
- **Immutable directories, once the path is ruled.** If the owner confirms the proposed
  `/data/opengwasdb/ld-stores/`, the served copy and any public copy are the same immutable
  directories: each is checked by `ogld.validate` after it is written, and a store is
  republished whole, under a new name, never edited in place. Until then the stores are at
  `/data/opengwasdb/work/ogld/stores/`, which is not that: the final lane replaces it by a swap.
  Before a public copy is announced, `opengwasdb.ld.open` on its URL runs the realdata reads: zarr reads a
  shard's index as a suffix range (§10).

## Open items this record does not settle

- **The rsID → ALID map ieugwasr needs.** ogld finds variants by ALID; ieugwasr sends rsIDs. No
  panel-wide map exists in ogld, the engine or ogdb-api; ogdb-api reads rsIDs from each engine
  Store's variant table.
- **ieugwasr's `ld_clump` default is refused by every HGDP+1kGP store.** r² 0.001 within 10 Mb is
  below every store's t_far (0.029–0.063), so it is a Refusal (ADR 0087). ogld-api passes it on;
  which configuration each ancestry offers by default is opengwasdb#416's.
- **Whether UK Biobank's LD Store (ogld#10) may be public as static Zarr.** This record covers
  the five HGDP+1kGP stores, whose genotypes are public (ADR 0088, decision 7).
- **The public host,** and with it when decision 2 takes effect. The options are above.
- **The served copy's directory:** `/data/opengwasdb/ld-stores/`, as proposed above, until the
  owner confirms it or names another.

## What this ticket could not find

- Whether the IEU node can serve the public internet, and the University's egress terms.
- OCI's current egress terms: Oracle's pricing pages refused the fetch, and third-party
  summaries disagree on whether the 10 TB allowance still applies.
- A real WAN: every HTTP number is loopback with injected latency.
- What each route costs in people's time; the work each needs is listed, not estimated.
