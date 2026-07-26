import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import {
  getSearchIndex,
  setVectorIndex,
  getVectorIndex,
  setEmbeddingProvider,
  removeSupersededFromIndex,
  indexMemory,
} from "../src/functions/search.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { memoryToObservation } from "../src/state/memory-utils.js";
import type { Memory, EmbeddingProvider } from "../src/types.js";

function makeMemory(overrides: Partial<Memory> = {}): Memory {
  return {
    id: "mem_idx_1",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    type: "pattern",
    title: "Unique widget frobnicator pattern",
    content: "Describes how the widget frobnicator caches results",
    concepts: ["widget"],
    files: [],
    sessionIds: ["ses_1"],
    strength: 5,
    version: 1,
    isLatest: true,
    ...overrides,
  };
}

function makeEmbeddingProvider(dimensions = 4): EmbeddingProvider {
  return {
    name: "mock",
    dimensions,
    embed: async () => new Float32Array(dimensions).fill(0.5),
    embedBatch: async (texts) =>
      texts.map(() => new Float32Array(dimensions).fill(0.5)),
  };
}

describe("search index maintenance helpers", () => {
  beforeEach(() => {
    setVectorIndex(new VectorIndex());
    setEmbeddingProvider(makeEmbeddingProvider());
  });

  it("removeSupersededFromIndex removes the memory from BM25 and the vector index", async () => {
    const mem = makeMemory({ id: "mem_remove_1" });
    getSearchIndex().add(memoryToObservation(mem));
    await indexMemory(mem);

    expect(getSearchIndex().has(mem.id)).toBe(true);
    expect(getVectorIndex()!.size).toBeGreaterThan(0);

    removeSupersededFromIndex(mem.id);

    expect(getSearchIndex().has(mem.id)).toBe(false);
    const bm25Results = getSearchIndex().search("frobnicator", 10);
    expect(bm25Results.find((r) => r.obsId === mem.id)).toBeUndefined();
    expect(getVectorIndex()!.search(new Float32Array(4).fill(0.5), 10)
      .find((r) => r.obsId === mem.id)).toBeUndefined();
  });

  it("indexMemory makes a memory findable in BM25", async () => {
    const mem = makeMemory({
      id: "mem_index_1",
      title: "Zebra quantum flux capacitor design",
      content: "Notes on the zebra quantum flux capacitor design",
    });

    expect(getSearchIndex().has(mem.id)).toBe(false);

    await indexMemory(mem);

    expect(getSearchIndex().has(mem.id)).toBe(true);
    const results = getSearchIndex().search("quantum flux capacitor", 10);
    expect(results.find((r) => r.obsId === mem.id)).toBeDefined();
  });
});
