import { it, describe } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  type DataValue,
  type InternalProcessContext,
  globalRivetNodeRegistry,
  registerIntegration,
  SemanticCacheNodeImpl,
  type SemanticCacheNode,
  cosineSimilarity,
  selectCacheHit,
} from '../../../src/index.js';

/* eslint-disable @typescript-eslint/no-floating-promises */

const createNode = (data: Partial<SemanticCacheNode['data']>) =>
  new SemanticCacheNodeImpl({
    ...SemanticCacheNodeImpl.create(),
    data: {
      ...SemanticCacheNodeImpl.create().data,
      embeddingIntegration: 'test',
      vectorDbIntegration: 'test',
      collectionId: 'test-collection',
      similarityThreshold: 0.9,
      ...data,
    },
  });

const ctx = {} as InternalProcessContext;

// Register mock integrations that the node composes via getIntegration(). The
// embedding generator and vector DB are controllable per-test through these
// module-level handles so we can drive cache-hit and cache-miss paths.
let embeddingFor: (text: string) => number[];
let nearestResult: DataValue;
const stored: { vector: number[]; data: unknown }[] = [];

registerIntegration('embeddingGenerator', 'test', () => ({
  generateEmbedding: async (text: string) => embeddingFor(text),
}));

registerIntegration('vectorDatabase', 'test', () => ({
  store: async (_collection: DataValue, vector: { value: number[] }, data: unknown) => {
    stored.push({ vector: vector.value, data });
  },
  nearestNeighbors: async () => nearestResult,
}));

const entry = (response: string, embedding: number[]): DataValue => ({
  type: 'object[]',
  value: [{ id: 'x', data: undefined, metadata: { response, embedding } }],
});

describe('SemanticCacheNode wiring', () => {
  it('is registered in the built-in node registry', () => {
    // Imports the (non-new) registry module and proves the Nodes.ts wiring landed.
    assert.ok(globalRivetNodeRegistry.isRegistered('semanticCache' as never));
  });

  it('creates a semanticCache node', () => {
    const node = SemanticCacheNodeImpl.create();
    assert.strictEqual(node.type, 'semanticCache');
  });
});

describe('SemanticCacheNode.process', () => {
  it('returns the cached response on a similarity hit', async () => {
    const cachedEmbedding = [1, 0, 0];
    embeddingFor = () => cachedEmbedding;
    nearestResult = entry('cached answer', cachedEmbedding);
    stored.length = 0;

    const node = createNode({ similarityThreshold: 0.9 });
    const inputs: Record<string, DataValue> = {
      prompt: { type: 'string', value: 'what is your return policy?' },
    };

    const result = await node.process(inputs, ctx);

    assert.strictEqual(result['hit' as never]!.value, true);
    assert.strictEqual(result['response' as never]!.value, 'cached answer');
    assert.strictEqual(result['similarity' as never]!.value, 1);
    // A hit must not re-write to the cache.
    assert.strictEqual(stored.length, 0);
  });

  it('writes the wired response back to the cache on a miss', async () => {
    const queryEmbedding = [1, 0, 0];
    embeddingFor = () => queryEmbedding;
    // A dissimilar cached prompt -> cosine similarity 0, below the threshold.
    nearestResult = entry('unrelated', [0, 1, 0]);
    stored.length = 0;

    const node = createNode({ similarityThreshold: 0.9 });
    const inputs: Record<string, DataValue> = {
      prompt: { type: 'string', value: 'how do I reset my password?' },
      response: { type: 'string', value: 'fresh model answer' },
    };

    const result = await node.process(inputs, ctx);

    assert.strictEqual(result['hit' as never]!.value, false);
    assert.strictEqual(result['response' as never]!.value, 'fresh model answer');
    // The new prompt+response pair is persisted with its embedding.
    assert.strictEqual(stored.length, 1);
    assert.deepStrictEqual(stored[0]!.vector, queryEmbedding);
    assert.deepStrictEqual((stored[0]!.data as { value: { response: string } }).value.response, 'fresh model answer');
  });

  it('passes the prompt through when no response is wired on a miss', async () => {
    embeddingFor = () => [1, 0, 0];
    nearestResult = { type: 'object[]', value: [] };
    stored.length = 0;

    const node = createNode({ similarityThreshold: 0.9 });
    const inputs: Record<string, DataValue> = {
      prompt: { type: 'string', value: 'brand new question' },
    };

    const result = await node.process(inputs, ctx);

    assert.strictEqual(result['hit' as never]!.value, false);
    assert.strictEqual(result['response' as never]!.value, 'brand new question');
    // Nothing to cache without a completed response.
    assert.strictEqual(stored.length, 0);
  });
});

describe('SemanticCache pure helpers', () => {
  it('computes cosine similarity', () => {
    assert.strictEqual(cosineSimilarity([1, 0, 0], [1, 0, 0]), 1);
    assert.strictEqual(cosineSimilarity([1, 0, 0], [0, 1, 0]), 0);
    assert.strictEqual(cosineSimilarity([1, 2, 3], [2, 4, 6]), 1);
    assert.ok(Math.abs(cosineSimilarity([1, 0], [1, 1]) - Math.SQRT1_2) < 1e-9);
    // Degenerate inputs return 0 rather than NaN.
    assert.strictEqual(cosineSimilarity([], []), 0);
    assert.strictEqual(cosineSimilarity([0, 0], [0, 0]), 0);
  });

  it('selects a cache hit only above the threshold', () => {
    const query = [1, 0, 0];
    const entries = [
      { response: 'a', embedding: [1, 0, 0] },
      { response: 'b', embedding: [0, 1, 0] },
    ];

    assert.deepStrictEqual(selectCacheHit(query, entries, 0.9), { hit: true, similarity: 1, response: 'a' });
    assert.deepStrictEqual(selectCacheHit(query, entries, 1.0001), { hit: false, similarity: 1, response: null });
    assert.deepStrictEqual(selectCacheHit(query, [], 0.9), { hit: false, similarity: 0, response: null });
  });
});
