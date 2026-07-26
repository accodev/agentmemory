import { describe, it, expect, beforeEach } from "vitest";
import { HybridSearch } from "../src/state/hybrid-search.js";
import { SearchIndex } from "../src/state/search-index.js";
import type { CompressedObservation, EmbeddingProvider } from "../src/types.js";

function makeObs(
  overrides: Partial<CompressedObservation> = {},
): CompressedObservation {
  return {
    id: "obs_1",
    sessionId: "ses_1",
    timestamp: new Date().toISOString(),
    type: "file_edit",
    title: "Edit auth middleware",
    subtitle: "JWT validation",
    facts: ["Added token check"],
    narrative: "Modified the auth middleware to validate JWT tokens",
    concepts: ["authentication", "jwt"],
    files: ["src/middleware/auth.ts"],
    importance: 7,
    ...overrides,
  };
}

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    get: async <T>(scope: string, key: string): Promise<T | null> => {
      return (store.get(scope)?.get(key) as T) ?? null;
    },
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

describe("HybridSearch", () => {
  let bm25: SearchIndex;
  let kv: ReturnType<typeof mockKV>;

  beforeEach(() => {
    bm25 = new SearchIndex();
    kv = mockKV();
  });

  it("returns BM25-only results when no vector index is provided", async () => {
    const obs = makeObs({ id: "obs_1", sessionId: "ses_1" });
    bm25.add(obs);
    await kv.set("mem:obs:ses_1", "obs_1", obs);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("auth");

    expect(results.length).toBe(1);
    expect(results[0].observation.id).toBe("obs_1");
    expect(results[0].vectorScore).toBe(0);
    expect(results[0].bm25Score).toBeGreaterThan(0);
  });

  it("returns empty results for no-match query", async () => {
    const obs = makeObs({ id: "obs_1", sessionId: "ses_1" });
    bm25.add(obs);
    await kv.set("mem:obs:ses_1", "obs_1", obs);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("database");
    expect(results).toEqual([]);
  });

  it("combinedScore is derived from bm25Score when no vector index", async () => {
    const obs = makeObs({ id: "obs_1", sessionId: "ses_1" });
    bm25.add(obs);
    await kv.set("mem:obs:ses_1", "obs_1", obs);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("auth");

    expect(results[0].combinedScore).toBeGreaterThan(0);
    expect(results[0].vectorScore).toBe(0);
    expect(results[0].graphScore).toBe(0);
  });

  it("results are sorted by combinedScore descending", async () => {
    const obs1 = makeObs({
      id: "obs_1",
      sessionId: "ses_1",
      title: "auth handler",
      narrative: "auth auth auth module",
      concepts: ["auth"],
    });
    const obs2 = makeObs({
      id: "obs_2",
      sessionId: "ses_1",
      title: "database setup",
      narrative: "auth connection config",
      concepts: ["database"],
    });
    bm25.add(obs1);
    bm25.add(obs2);
    await kv.set("mem:obs:ses_1", "obs_1", obs1);
    await kv.set("mem:obs:ses_1", "obs_2", obs2);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("auth");

    expect(results.length).toBe(2);
    expect(results[0].combinedScore).toBeGreaterThanOrEqual(
      results[1].combinedScore,
    );
  });

  it("respects limit parameter", async () => {
    for (let i = 0; i < 10; i++) {
      const obs = makeObs({
        id: `obs_${i}`,
        sessionId: "ses_1",
        title: `auth feature ${i}`,
      });
      bm25.add(obs);
      await kv.set("mem:obs:ses_1", `obs_${i}`, obs);
    }

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("auth", 3);
    expect(results.length).toBe(3);
  });

  it("skips observations not found in KV", async () => {
    const obs = makeObs({ id: "obs_missing", sessionId: "ses_1" });
    bm25.add(obs);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("auth");
    expect(results).toEqual([]);
  });

  it("falls back to KV.memories when an indexed entry is a saved memory (#265)", async () => {
    // mem::remember writes to KV.memories under the synthetic sessionId
    // "memory" — the BM25 index sees that synthetic sessionId, but
    // KV.observations("memory") never has anything.
    const indexable = makeObs({
      id: "mem_abc",
      sessionId: "memory",
      title: "Test memory for search",
      narrative: "Test memory for search",
      concepts: ["test", "search"],
    });
    bm25.add(indexable);

    const memory = {
      id: "mem_abc",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      type: "fact",
      title: "Test memory for search",
      content: "Test memory for search",
      concepts: ["test", "search"],
      files: [],
      sessionIds: [],
      strength: 7,
      version: 1,
      isLatest: true,
    };
    await kv.set("mem:memories", "mem_abc", memory);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("test memory search");

    expect(results.length).toBe(1);
    expect(results[0].observation.id).toBe("mem_abc");
    expect(results[0].observation.narrative).toBe("Test memory for search");
    expect(results[0].observation.concepts).toEqual(["test", "search"]);
  });

  function makeMemory(id: string) {
    return {
      id,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      type: "fact",
      title: "Widget rollout status",
      content: "Widget rollout status",
      concepts: ["widget"],
      files: [],
      sessionIds: [],
      strength: 7,
      version: 1,
      isLatest: true,
    };
  }

  describe("curated-memory diversity", () => {
    it("scopes the diversity cap: mem_* dominate top results under scope=observations but not scope=all", async () => {
      // 10 curated memories, all synthetic sessionId "memory"
      for (let i = 0; i < 10; i++) {
        const id = `mem_${i}`;
        const doc = makeObs({
          id,
          sessionId: "memory",
          title: "Widget rollout status",
          narrative: "widget widget widget rollout status",
          concepts: ["widget"],
        });
        bm25.add(doc);
        await kv.set("mem:memories", id, makeMemory(id));
      }

      // 10 observations spread across 5 real sessions (2 each)
      for (let i = 0; i < 10; i++) {
        const id = `obs_${i}`;
        const sessionId = `ses_${i % 5}`;
        const doc = makeObs({
          id,
          sessionId,
          title: "widget note",
          narrative: "widget mention",
          concepts: ["widget"],
        });
        bm25.add(doc);
        await kv.set(`mem:obs:${sessionId}`, id, doc);
      }

      const observationsScope = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "observations", maxPerSession: 3 },
      );
      const resultsObsScope = await observationsScope.search("widget", 20);
      expect(
        resultsObsScope.slice(0, 5).every((r) => r.observation.id.startsWith("mem_")),
      ).toBe(true);

      const allScope = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "all", maxPerSession: 3 },
      );
      const resultsAllScope = await allScope.search("widget", 20);
      expect(
        resultsAllScope.slice(0, 5).every((r) => r.observation.id.startsWith("mem_")),
      ).toBe(false);
    });

    it("still caps real sessions under the default scope", async () => {
      // one bursty real session with 10 matching observations
      for (let i = 0; i < 10; i++) {
        const id = `obs_burst_${i}`;
        const doc = makeObs({
          id,
          sessionId: "ses_burst",
          title: "gadget note",
          narrative: "gadget gadget gadget",
          concepts: ["gadget"],
        });
        bm25.add(doc);
        await kv.set("mem:obs:ses_burst", id, doc);
      }

      // 20 more matching observations, one per distinct session, so the
      // main diversify pass fills the retrieval depth without backfill.
      for (let i = 0; i < 20; i++) {
        const id = `obs_other_${i}`;
        const sessionId = `ses_other_${i}`;
        const doc = makeObs({
          id,
          sessionId,
          title: "gadget note",
          narrative: "gadget mention",
          concepts: ["gadget"],
        });
        bm25.add(doc);
        await kv.set(`mem:obs:${sessionId}`, id, doc);
      }

      const hybrid = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "observations", maxPerSession: 3 },
      );
      const results = await hybrid.search("gadget", 20);
      const burstCount = results.filter((r) =>
        r.observation.sessionId === "ses_burst",
      ).length;
      expect(burstCount).toBeLessThanOrEqual(3);
    });

    it("honors a configured maxPerSession other than the historical default of 3", async () => {
      for (let i = 0; i < 10; i++) {
        const id = `obs_burst2_${i}`;
        const doc = makeObs({
          id,
          sessionId: "ses_burst2",
          title: "sprocket note",
          narrative: "sprocket sprocket sprocket",
          concepts: ["sprocket"],
        });
        bm25.add(doc);
        await kv.set("mem:obs:ses_burst2", id, doc);
      }
      for (let i = 0; i < 20; i++) {
        const id = `obs_other2_${i}`;
        const sessionId = `ses_other2_${i}`;
        const doc = makeObs({
          id,
          sessionId,
          title: "sprocket note",
          narrative: "sprocket mention",
          concepts: ["sprocket"],
        });
        bm25.add(doc);
        await kv.set(`mem:obs:${sessionId}`, id, doc);
      }

      const hybrid = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "observations", maxPerSession: 1 },
      );
      const results = await hybrid.search("sprocket", 20);
      const burstCount = results.filter((r) =>
        r.observation.sessionId === "ses_burst2",
      ).length;
      expect(burstCount).toBeLessThanOrEqual(1);
    });
  });

  describe("curated-memory type prior", () => {
    it("flips ranking based on obsPenalty when scores are close", async () => {
      // obs_ ranks first (rank 1) by term frequency; mem_ ranks second
      const obsDoc = makeObs({
        id: "obs_lead",
        sessionId: "ses_1",
        title: "sprocket status",
        narrative: "sprocket sprocket sprocket sprocket status",
        concepts: ["sprocket"],
      });
      const memDoc = makeObs({
        id: "mem_lead",
        sessionId: "memory",
        title: "sprocket status",
        narrative: "sprocket status",
        concepts: ["sprocket"],
      });
      bm25.add(obsDoc);
      bm25.add(memDoc);
      await kv.set("mem:obs:ses_1", "obs_lead", obsDoc);
      await kv.set("mem:memories", "mem_lead", makeMemory("mem_lead"));

      const noPenalty = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "observations", maxPerSession: 3 },
      );
      const noPenaltyResults = await noPenalty.search("sprocket status");
      expect(noPenaltyResults[0].observation.id).toBe("obs_lead");

      const withPenalty = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0.15, diversityScope: "observations", maxPerSession: 3 },
      );
      const withPenaltyResults = await withPenalty.search("sprocket status");
      expect(withPenaltyResults[0].observation.id).toBe("mem_lead");
    });

    it("does not hard-exclude obs_ results when they decisively outrank mem_", async () => {
      const obsDoc = makeObs({
        id: "obs_strong",
        sessionId: "ses_1",
        title: "widget status widget status widget",
        narrative: "widget widget widget widget widget status update report",
        concepts: ["widget"],
      });
      // 12 decoys that each beat mem_weak's rank, so mem_weak lands far
      // enough down the bm25 ranking that a 15% penalty on rank-1 obs_strong
      // still cannot let mem_weak overtake it.
      for (let i = 0; i < 12; i++) {
        const id = `obs_decoy_${i}`;
        const sessionId = `ses_decoy_${i}`;
        const decoy = makeObs({
          id,
          sessionId,
          title: "widget status",
          narrative: "widget status seen here",
          concepts: ["widget"],
        });
        bm25.add(decoy);
        await kv.set(`mem:obs:${sessionId}`, id, decoy);
      }
      const memDoc = makeObs({
        id: "mem_weak",
        sessionId: "memory",
        title: "unrelated note",
        narrative: "a passing widget mention",
        concepts: ["widget"],
      });
      bm25.add(obsDoc);
      bm25.add(memDoc);
      await kv.set("mem:obs:ses_1", "obs_strong", obsDoc);
      await kv.set("mem:memories", "mem_weak", makeMemory("mem_weak"));

      const hybrid = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0.15, diversityScope: "observations", maxPerSession: 3 },
      );
      const results = await hybrid.search("widget status");
      expect(results[0].observation.id).toBe("obs_strong");
      expect(results.some((r) => r.observation.id === "obs_strong")).toBe(true);
    });

    it("does not penalize non-obs_ prefixes", async () => {
      const customDoc = makeObs({
        id: "custom_1",
        sessionId: "ses_1",
        title: "trinket note",
        narrative: "trinket trinket status",
        concepts: ["trinket"],
      });
      bm25.add(customDoc);
      await kv.set("mem:obs:ses_1", "custom_1", customDoc);

      const noPenalty = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "observations", maxPerSession: 3 },
      );
      const noPenaltyResults = await noPenalty.search("trinket status");

      const withPenalty = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0.15, diversityScope: "observations", maxPerSession: 3 },
      );
      const withPenaltyResults = await withPenalty.search("trinket status");

      expect(withPenaltyResults[0].combinedScore).toBe(
        noPenaltyResults[0].combinedScore,
      );
    });
  });

  describe("superseded memory filtering", () => {
    function makeMemoryOverride(
      id: string,
      overrides: Record<string, unknown> = {},
    ) {
      return { ...makeMemory(id), ...overrides };
    }

    it("excludes a superseded (isLatest: false) row even when it lexically outranks its replacement", async () => {
      const superseded = makeObs({
        id: "mem_old",
        sessionId: "memory",
        title: "beacon status beacon status",
        narrative: "beacon beacon beacon beacon status status status",
        concepts: ["beacon"],
      });
      const latest = makeObs({
        id: "mem_new",
        sessionId: "memory",
        title: "beacon status",
        narrative: "beacon status",
        concepts: ["beacon"],
      });
      bm25.add(superseded);
      bm25.add(latest);
      await kv.set(
        "mem:memories",
        "mem_old",
        makeMemoryOverride("mem_old", { isLatest: false }),
      );
      await kv.set(
        "mem:memories",
        "mem_new",
        makeMemoryOverride("mem_new", { isLatest: true }),
      );

      const hybrid = new HybridSearch(bm25, null, null, kv as never);
      const results = await hybrid.search("beacon status");

      expect(results.some((r) => r.observation.id === "mem_old")).toBe(false);
      expect(results.some((r) => r.observation.id === "mem_new")).toBe(true);
    });

    it("still surfaces legacy rows missing the isLatest field", async () => {
      const legacy = makeObs({
        id: "mem_legacy",
        sessionId: "memory",
        title: "gizmo status",
        narrative: "gizmo status",
        concepts: ["gizmo"],
      });
      bm25.add(legacy);
      const legacyMemory: Record<string, unknown> = makeMemory("mem_legacy");
      delete legacyMemory.isLatest;
      await kv.set("mem:memories", "mem_legacy", legacyMemory);

      const hybrid = new HybridSearch(bm25, null, null, kv as never);
      const results = await hybrid.search("gizmo status");

      expect(results.some((r) => r.observation.id === "mem_legacy")).toBe(true);
    });

    it("leaves isLatest: true rows unaffected", async () => {
      const current = makeObs({
        id: "mem_current",
        sessionId: "memory",
        title: "widget status",
        narrative: "widget status",
        concepts: ["widget"],
      });
      bm25.add(current);
      await kv.set(
        "mem:memories",
        "mem_current",
        makeMemoryOverride("mem_current", { isLatest: true }),
      );

      const hybrid = new HybridSearch(bm25, null, null, kv as never);
      const results = await hybrid.search("widget status");

      expect(results.some((r) => r.observation.id === "mem_current")).toBe(true);
    });
  });

  describe("upstream equivalence escape hatch", () => {
    it("reproduces exact RRF scores with obsPenalty=0, scope=all", async () => {
      const docA = makeObs({
        id: "obs_a",
        sessionId: "ses_1",
        title: "beacon status report",
        narrative: "beacon beacon beacon beacon status report update",
        concepts: ["beacon"],
      });
      const docB = makeObs({
        id: "obs_b",
        sessionId: "ses_2",
        title: "beacon status",
        narrative: "beacon beacon status update",
        concepts: ["beacon"],
      });
      const docC = makeObs({
        id: "obs_c",
        sessionId: "ses_3",
        title: "beacon mention",
        narrative: "beacon appears once here",
        concepts: ["beacon"],
      });
      bm25.add(docA);
      bm25.add(docB);
      bm25.add(docC);
      await kv.set("mem:obs:ses_1", "obs_a", docA);
      await kv.set("mem:obs:ses_2", "obs_b", docB);
      await kv.set("mem:obs:ses_3", "obs_c", docC);

      const hybrid = new HybridSearch(
        bm25,
        null,
        null,
        kv as never,
        undefined,
        undefined,
        undefined,
        undefined,
        { obsPenalty: 0, diversityScope: "all", maxPerSession: 3 },
      );
      const results = await hybrid.search("beacon status");

      // Only bm25 leg is active, so effectiveBm25W === 1 and
      // combinedScore === 1 / (60 + rank) for each result's bm25 rank.
      const RRF_K = 60;
      results.forEach((r, i) => {
        expect(r.combinedScore).toBeCloseTo(1 / (RRF_K + (i + 1)), 10);
      });
    });
  });
});
