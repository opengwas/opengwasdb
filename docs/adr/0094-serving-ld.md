# LD is served by ogdb-api's job workers from local disk, and the LD Stores are also published as static Zarr

**Status: Proposed.** Nothing here is decided. The owner decides all three questions below; this
record gives the options, what each costs and a recommendation for each, and becomes the decision
when the owner has ruled, with their ruling recorded under "Who decided".

Issue [#9](https://github.com/opengwas/ogld/issues/9), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
The reads are in `docs/spec/ld-store.md` §10 and their compatibility in §13. The measurements
are in `docs/benchmarks/latency.md` (ogld#8), `docs/benchmarks/fast-reader.md` (ogld#32),
`docs/benchmarks/far-pairs.md` (ogld#31) and `docs/benchmarks/serving.md` (this ticket), each
rendered from records under `docs/benchmarks/results/`. The terms are in `CONTEXT.md`.

**Who decided.** No one yet. The owner rules on ogld#9, and the ruling (who, when, where) goes
here, with the status changed to Accepted and any option the owner chose instead written in.

## The questions

1. **The serving route.** (a) ogdb-api runs `ogld` as a library in its job workers, which
   already run queries as jobs; or (b) a separate service in `opengwas/ogld-api`.
2. **Static Zarr.** Whether the LD Stores are also served as static Zarr that clients read
   themselves with `ogld.open("https://…")`.
3. **Where the five HGDP+1kGP LD Stores are published.** This moved here from ogld#7 (owner,
   11 November 2026). They are staged at `/data/opengwasdb/work/ogld/stores/`.

## Context

**The stores.** The five stores of ogld#7 hold 533.4 GB in 58,123 files: Near Bands 486.1 GB,
Proxy Indexes 45.2 GB, Far Pairs 1.46 GB (ogld#31) and variant tables 0.56 GB, over 47.2 M
variants. AFR alone is 241.2 GB in 17,009 files. Each is immutable once validated, format
`0.1.0` (§13), layout C (ADR 0006).

**ogdb-api as it is** (opengwas/ogdb-api at `3f70e0e`, read on 13 November 2026):

- **Where it runs.** A Compose deployment, an `api` container and Redis, with `/data/opengwasdb`
  mounted read-only. Its owner's instance, port 6433, runs on the IEU node that holds
  `/data/opengwasdb` and the staged LD Stores, and reads the engine's Stores from local disk; its
  agents' VM runs on the same node and mounts `/data` over SSHFS. On 13 November that instance
  served 13 Stores with `backend_max_running` 180, in one uvicorn process of 316 MB resident.
  Its AGENTS.md places "the main live server" elsewhere (below, "could not find").
- **How a query runs.** Every data query is a job keyed by (user, canonical query hash) in a
  Redis queue (its #11). Each API process runs **one** worker coroutine, which claims one job at
  a time and runs its store query on a bounded thread pool in the same process
  (`ogdb_api/operations.py`); more throughput means more API processes on one Redis, whose
  admission is shared (its #18). Its stress run measured 0.2 jobs/s for one process, and "latency
  grows with queue depth, not with memory" (DEVLOG, its #10).
- **Limits a job meets.** A store query that runs past `OGDB_STORE_TIMEOUT_SECONDS` (30 s) fails
  terminally; the query cannot be interrupted, so its thread and release slot stay held until it
  returns (`ogdb_api/planner.py`, `BackendLimiter`).
- **How a result is returned.** The client polls `GET /api/jobs/{id}` at the interval the API
  recommends, then reads the result: inline up to 10,000 rows, else immutable NDJSON pages, at
  most 100,000 rows and 50 MB a job, held in a results cache of 1 GiB until every page is read
  plus 300 s (its #16).
- **Memory.** Its specification says memory, not CPU, is the binding constraint: OGS-00011's
  reader holds about 24 GiB, and on a 32 GiB VM the whole deployment was OOM-killed at 30.9 GiB.
- **LD today.** `/api/ld/clump`, `/api/ld/matrix` and `/api/ld/reflookup`, OpenGWAS's own LD
  paths, are registered and answer `501 Not yet implemented` until "an explicitly supported LD
  panel and algorithm" exists (its SPECIFICATION §5). Its ieugwasr and ieugwaspy compatibility
  epic (its #35) keeps those clients' paths in ogdb-api.
- **Versioning.** It pins `opengwasdb` to one Git revision and upgrades it deliberately, by a
  ten-step procedure in its README.
- **In front of it.** opengwas/opengwas-gateway authenticates users (JWT) and forwards
  `/{namespace}/{path}` to one backend per namespace, `ogdb` the only one so far.

**The engine.** Engine issue opengwas/opengwasdb#419 makes the engine's own Top-Hit Query clump
on the fly with `ogld.clump`, against an LD Store the caller names, when the configuration is not
one of the Store's stored Clumps (#418). ogdb-api serves that query, so under either route ogdb-api
imports ogld and reads LD Stores for top hits. The engine's roadmap holds that a store "should
support laptop, HPC, and cloud/API usage" and that "the service layer should route and authorise
queries, not become the only place the data can be used" (`docs/opengwas-roadmap.md`, principle
2), and that workers may "mount stores from local disk, object storage, or HPC filesystems"
(Horizon 4). It has no ADR on serving over HTTP or on where Stores are hosted: its ADR 0018 puts
remote querying and store discovery in "a separate catalogue or deployment layer", and its Stores
live under `/data/opengwasdb/stores/OGS-*` on the same node, listed by opengwasdb-stores' Analysis
Catalog (its ADR 0039), which records no path.

**What the reads cost**, on the published EUR store, from local disk and over HTTP with a wait
added to every request. "Now" is the reader with ogld#31's Far Pairs and ogld#32's changes, timed
by this ticket (`docs/benchmarks/serving.md`); ogld#31 changed no clump or proxy read.

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

## Decision proposed

### 1. The route: (a), ogdb-api's job workers, with LD jobs in worker processes of their own

LD queries (`clump`, `matrix`, `region`, `proxies`, and `reflookup` as a presence lookup) become
job operations of ogdb-api, submitted, polled and returned through its job contract like every
other data query, and served at its existing `/api/ld/*` paths. They run in ogdb-api processes
that claim only LD jobs: the same image, Redis and admission, with their own timeout and
memory budget. Each such process opens the LD Stores once, at startup, and keeps them (§10).

- **Why ogdb-api.** The job queue, its keys, polling, retention, queue controls, metrics,
  admission and the gateway's route all exist there; ogld-api would rebuild each one (the
  repository holds one commit, a README). The paths clients call for LD are already
  ogdb-api's, and ogdb-api imports ogld for top-hit clumps under either route
  (opengwasdb#419).
- **Why LD workers of their own.** Under ogdb-api as it is, an LD job would share a process with
  association jobs, and that costs three things, measured here or read from its code:
  - _the queue_: one worker a process runs one job at a time, so a clump, 6.3 s at the median
    and up to 69 s, holds up every association job behind it;
  - _the timeout_: 4 of #403's 37 Analyses clump past ogdb-api's 30 s at r² 0.01, the largest
    (260,821 hits) in 69 s, and 1 at r² 0.001;
  - _memory_: a process with the five stores open holds 1.45 GiB of chromosome arrays once it
    has met every chromosome, and 21.5 GiB resident once their 4 GiB caches fill (peak
    22.1 GiB). ogdb-api's 32 GiB VM was OOM-killed at 30.9 GiB without them.

  An LD worker process removes all three without a second service: an LD OOM ends that process,
  its job's lease lapses and the reaper requeues it (ogdb-api's SPECIFICATION §7.2), and
  association jobs never wait behind a clump. Two other costs of sharing a process were measured
  and are small. While ogld reads on a worker thread, an asyncio loop on the main thread wakes
  1–16 ms late at the 99th percentile and at most 116 ms late, against 0.4 ms idle. And under the
  zarr codec pipeline the engine sets process-wide when imported (opengwasdb ADR 0056), a clump
  and a matrix took within the run-to-run spread (−7% to +15%) and a 3 Mb region 15–17% longer.
- **Latency does not choose between (a) and (b).** Both run on the node that holds the stores
  and read them from local disk. What latency separates is the server routes from static Zarr
  (decision 2).

### 2. Static Zarr: yes, for clients that are near the data or need whole stores

The same immutable store directories are published where a static host serves them with byte
ranges, and `ogld.open("https://…")` reads them as it reads a local store (§10: "a static host
that honours byte ranges is enough"). This needs no code: the reader, its refusals and its
answers are the same over HTTP, and ogld#8, #31 and #32 measured them.

It is the route for clients near the data (a cluster in the same data centre, a copy on HPC
storage) and for reproducing a result against a named store. It is not the interactive route for
a remote user: every batch of chunks waits a round trip, so a 500-variant genome-wide matrix is
8.3 s locally, 36.7 s at 50 ms and 61.2 s at 100 ms, a third of it loading the chromosomes'
arrays at those latencies; and a clump reads Near Band chunks by the gigabyte, 48.6 GB for the
37 Analyses, which a server reads from its own disk.

### 3. Where: the IEU node for ogdb-api, and an object store with no egress fee for static Zarr

- **The served copy** is on local disk beside ogdb-api's LD workers. Today that is `/data` on
  the IEU node, where ogdb-api's owner's instance runs: the stores move from the work area to a
  read-only release root beside the engine's Stores, 533 GB of the volume's 18 TB free. If
  ogdb-api's live server runs elsewhere (below, "could not find"), a copy goes there, since
  reading over a network would cost a clump job 4.3× its local time at 50 ms (ogld#32's four
  Analyses: 224 s against 52 s).
- **The public static copy** is uploaded to object storage that charges nothing for egress:
  Cloudflare R2, at a list price of $0.015 per GB-month, about $8 a month for 533 GB, and
  $0.36 a million reads. If OpenGWAS's existing cloud account makes one vendor simpler, OCI
  Object Storage, at about $13.60 a month and with the 10 TB of free egress a month Oracle
  advertises (unconfirmed here; below, "could not find"), is the alternative.

No store moves and nothing is uploaded until the owner rules.

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
| top-hit clumps (opengwasdb#419) | the same process and pin | still ogld inside ogdb-api: two pins of ogld to keep in step |
| gateway | `ogdb`'s paths, unchanged | a second namespace, or ogdb-api forwarding `/api/ld/*` |
| to build | LD operations, an LD worker role, the readers kept open, a timeout per operation | a service, its deployment, and all of ogdb-api's job system again |

- **(a) as ogdb-api is, with LD jobs in its one worker.** Rejected: the three costs above.
- **(a) with LD workers of their own.** Proposed.
- **(b).** It isolates LD and versions it on its own, and both are what (a) gets from worker
  processes of their own except the separate pin and deploy. It costs a second job system and a
  second service to run, and it does not remove ogld from ogdb-api (opengwasdb#419). Choose it
  if LD must be released apart from ogdb-api, or if ogdb-api is not to carry LD at all.
- **Synchronous LD endpoints, no job.** Rejected: a clump can run for minutes, and ogdb-api's
  owner decided that every data query is a job and that direct routes go (6 November 2026, its
  #11 and #14), with ieugwasr's and ieugwaspy's compatibility paths the one exception that waits
  server-side, on a job.

### Static Zarr

- **Yes, public** (proposed). No code; the store's own refusals and NaN travel with it.
  Remote clients pay round trips and bytes, and the host pays egress unless it is free.
- **No: the API only.** It keeps every read behind authentication and in the usage logs, and
  no egress is paid, but a store can then be used only through the service, against the
  engine's principle 2, and a cluster user moves every result through polling.
- **Yes, behind authentication** (signed URLs, or a gateway in front of the static host). It
  keeps the usage record, at the cost of a token every client must obtain. Possible later for
  a store whose terms need it; the HGDP+1kGP stores do not (their genotypes are public under
  gnomAD's terms, ADR 0002 decision 7; ADR 0001 decision 5 allows LD Stores to be published).

### Where

List prices read on 13 November 2026 from the providers' pages or, where those would not load,
from third-party summaries, for 533 GB. "A full clump" is the 48.6 GB, in 86,286 requests, that
#403's 37 Analyses read at r² 0.01 (`docs/benchmarks/serving.md`).

| place | storage a month | egress | a full clump read remotely | access control | ogdb-api reads it locally |
|---|---|---|---|---|---|
| the IEU node, a static web server | none new (533 GB of 18 TB free) | the University's terms were not found | no charge found; bounded by the node's link | the web server's | yes |
| Cloudflare R2 | about $8 | free | about $0.03 of read requests | public bucket, or tokens | no |
| OCI Object Storage | about $13.60 | 10 TB a month free, then about $0.0085/GB | free within 10 TB | pre-authenticated requests, policies | no |
| AWS S3 Standard | $12.26 | $0.09/GB after 100 GB a month | about $4.40 | bucket policy, signed URLs | no |
| Backblaze B2 | about $3.70 | free to 3× stored a month (1.6 TB), then $0.01/GB | free within 1.6 TB | public bucket, or keys | no |
| a CDN in front | the origin's | CloudFront about $0.085/GB in Europe after 1 TB free; Cloudflare's CDN terms restrict large non-HTML files not hosted on R2 | the origin's, less cache hits | the CDN's | no |
| Zenodo | free | free | — | public | no |

- **The IEU node alone.** Free and already where ogdb-api reads, but it makes a compute node a
  public web host, which needs the University's network and security approval, and its
  durability is the node's backups. Rejected as the public copy; kept as the served copy.
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

- **ogdb-api carries LD.** Its LD operations need: a request schema for each `/api/ld/*` path; a
  refusal (§9) returned as a request error, kept as ogdb-api keeps them (its #17), never a
  clump at a smaller window; a cost model, so admission prices a clump by its hits; an LD worker
  role that claims only LD jobs; a timeout per operation; and `cache_bytes` sized to the worker's
  memory (§10).
- **ieugwasr's defaults are refused on these stores.** `ld_clump`'s default, r² 0.001 within
  10 Mb, is below every HGDP+1kGP store's t_far (0.029–0.063), so it is a Refusal (ADR 0001).
  Which configuration each ancestry gets by default is opengwasdb#416's; the API passes the
  Refusal on.
- **ALIDs, not rsIDs.** ogld finds variants by ALID. ieugwasr sends rsIDs, and no panel-wide rsID
  index exists in either repository; ogdb-api reads rsIDs from each engine Store's variant table.
  The LD operations need a mapping, which this record does not choose.
- **A reader change redeploys ogdb-api.** A new `format_version` or encoding is refused by a
  reader that does not implement it (§13), so a store in a new format waits for ogdb-api's ogld
  pin, as an engine Store waits for its opengwasdb pin. A chunk or shard change does not.
- **Two copies to keep in step.** The served and public copies are the same immutable
  directories; each is checked by `ogld.validate` after it is written, and a store is
  republished whole, under a new name, never edited in place.
- **The first upload is checked by reading it.** zarr reads a shard's index as a suffix range
  (§10). Before the public copy is announced, `ogld.open` on its URL runs the realdata reads.

## What this ticket could not find

- ogdb-api's production deployment: its AGENTS.md calls it "the main live server, elsewhere",
  not reachable from its VM. Whether it is on this node, on OCI (where the gateway's Redis is,
  by its configuration's key `oci`) or elsewhere decides whether its workers can read `/data`.
- Whether the IEU node can serve the public internet, and the University's egress terms.
- OCI's current egress terms: Oracle's pricing pages refused the fetch, and third-party
  summaries disagree on whether the 10 TB allowance still applies.
- A real WAN: every HTTP number is loopback with injected latency.
- What each route costs in people's time. The work each needs is listed above; no estimate in
  weeks is made here.
- Whether UK Biobank's LD Store (ogld#10) may be published as static Zarr. This record covers the
  five HGDP+1kGP stores.
