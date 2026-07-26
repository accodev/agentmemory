# Curated memories are hard-capped at 3 per query

Status: proposed · Branch: `feat/curated-memory-ranking` · Base: v0.9.28

## The bug

`src/functions/remember.ts:107` writes every memory with `sessionIds: []`. `src/state/memory-utils.ts:14`
then assigns them all the same synthetic session:

```ts
sessionId: memory.sessionIds?.[0] ?? "memory",
```

`diversifyBySession(..., maxPerSession = 3)` treats that synthetic bucket as if it were a real
session, so **at most 3 curated memories can be returned per query, regardless of score**. The cap
exists to stop one bursty tool-call session monopolising results; `"memory"` is not a session.

Measured on a 911-memory store with 21,971 competing observation rows (a 24:1 ratio), curated
memories occupied **38.5%** of result slots — about 3.2 per 10-result query, which is the cap plus
occasional backfill, not a ranking outcome.

Related open issues: #819 (no way to search only `memory_save` entries) and #993 (post-tool-use
observations polluting recall).

## The change

Two things, 19 changed lines in `hybrid-search.ts`.

1. **Scope the cap.** Apply it only when `AGENTMEMORY_DIVERSITY_SCOPE=all`, or to ids prefixed
   `obs_`. Observations remain capped per real session — that protection is preserved. Only rows the
   cap actually applies to consume the quota, so an exempt memory carrying a real `sessionId` cannot
   spend the budget belonging to that session's observations.
2. **Optional type prior.** Multiply the existing RRF score by `(1 - obsPenalty)` for `obs_` ids. The
   `obs_` prefix is penalised rather than `mem_` whitelisted, so other curated prefixes are never
   accidentally demoted. Multiplicative, therefore relevance-proportional: a decisively better
   observation still wins, and observations are never excluded.

The RRF expression itself is untouched, so scores keep their existing scale and there is no
interaction with `reranker.ts`'s per-item fallback.

| Env var | Range | Default |
|---|---|---|
| `AGENTMEMORY_OBS_PENALTY` | [0, 0.9] | 0.15 |
| `AGENTMEMORY_DIVERSITY_SCOPE` | `observations` \| `all` | `observations` |
| `AGENTMEMORY_MAX_PER_SESSION` | ≥ 1 | 3 |

`AGENTMEMORY_OBS_PENALTY=0` with `AGENTMEMORY_DIVERSITY_SCOPE=all` reproduces upstream behaviour
exactly, including identical `combinedScore` values.

## Measurement

100 probes generated mechanically from the corpus, ground truth fixed by rule before any retrieval
(a memory is relevant iff `title + content` contains the probe anchor, case-insensitive), committed
to a hash (`probes.lock`) and verified at every run. Four strata: 40 exact-identifier, 25
aggregation, 25 cross-project collision, 10 absent-fact. Two byte-identical 247 MB store clones
(385 files, 0 differing), same probes both arms, 0 request errors.

| | upstream | patched | delta |
|---|---|---|---|
| recall@1 | 22.2% | 24.4% | +2.2 |
| recall@5 | 25.6% | 25.6% | **+0.0** |
| recall@10 | 25.6% | **42.2%** | **+16.7** |
| MRR@10 | 0.234 | 0.268 | +0.034 |
| curated slot share | 38.5% | 96.6% | +58.1 |

Bootstrap 95% CI on the recall@10 delta (10,000 resamples, paired per-probe): **[8.9, 24.4]** — does
not cross zero. recall@5: **[0.0, 0.0]** — no effect.

## What did not work, and is therefore not in this patch

An earlier version of this work added magnitude-aware score fusion: per-arm divide-by-max
normalisation blended with RRF at `rankWeight = 0.7`, grounded in Fox & Shaw's CombSUM (TREC-2 1994)
and Lee's normalisation analysis (SIGIR 1997), with the min-subtraction half of min-max deliberately
dropped. It was ~176 lines with its own test suite and mutation coverage.

Measured on the same probe set, **turning it off scored better on every metric**: recall@1 24.4% vs
23.3%, recall@10 42.2% vs 37.8%, MRR 0.268 vs 0.256. Pure RRF ranks these queries better than the
blend. The entire gain came from the two changes above.

It is omitted rather than shipped disabled. Cormack et al. (SIGIR 2009) argue RRF's value is
robustness to incomparable score scales across arms; on this corpus that robustness appears to be
worth more than the magnitude information the blend recovers.

## Two other findings from the same experiment

**Raising `VECTOR_WEIGHT` is actively harmful here.** Restoring the upstream default of 0.6 (from a
local 0.15) collapsed retrieval to recall@1 7.8%, recall@5 10.0%, recall@10 22.2% — worse than
unpatched upstream. With ~22k observation rows, most queries have hundreds of semantically similar
tool-call logs, so the vector arm surfaces noise while BM25's lexical precision finds the specific
identifier. Corpora dominated by raw capture may want the vector arm demoted, not strengthened.

**recall@5 is a candidate-generation ceiling, not a ranking problem.** It sat at exactly 25.6% across
upstream and every patched configuration that did not actively regress. Nothing in ranking moved it.
BM25 fetches `limit * 2` candidates; for roughly three-quarters of probes the relevant memory never
enters the pool, so no fusion, prior or reranker can recover it. Improving the top 5 requires
retrieving deeper.

## Limitations

- Probe queries are templated from anchors present in the corpus, so vocabulary overlap between query
  and target is higher than in natural use. The bias applies equally to both arms, so the delta is
  more trustworthy than the absolute values.
- One of 100 anchors is a memory id rather than a code identifier.
- Paraphrase robustness is untested — every probe contains its anchor verbatim.
- `absentAnyResult` was 100% in every configuration: for all 10 probes naming a non-existent
  identifier, both upstream and patched returned confident results. Neither abstains. Out of scope
  here, but it means "no memory of this" is not currently expressible.
- The `RANK_WEIGHT` comparison that led to dropping the blend was one of four configurations tried
  against a single baseline, so it carries a multiple-comparison risk. The headline diversity-cap
  result does not: it was a single pre-registered comparison.

## Citations

- **Cormack, Clarke & Buettcher**, SIGIR (2009), *Reciprocal Rank Fusion Outperforms Condorcet and
  Individual Rank Learning Methods* — why RRF is left intact.
- **Fox & Shaw**, TREC-2 (1994) and **Lee**, SIGIR (1997), *Analyses of Multiple Evidence
  Combination* — the basis for the magnitude-blend approach that measurement rejected.
