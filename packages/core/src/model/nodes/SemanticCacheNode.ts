import {
  type ArrayDataValue,
  type ChartNode,
  type DataValue,
  type EditorDefinition,
  type Inputs,
  type InternalProcessContext,
  type NodeId,
  type NodeInputDefinition,
  type NodeOutputDefinition,
  type NodeUIData,
  type Outputs,
  type PortId,
  type ScalarDataValue,
  type VectorDataValue,
} from '../../index.js';
import { NodeImpl } from '../NodeImpl.js';
import { coerceType, coerceTypeOptional, dedent, getInputOrData, newId } from '../../utils/index.js';
import { nodeDefinition } from '../NodeDefinition.js';
import { getIntegration } from '../../integrations/integrations.js';

/**
 * Semantic Cache node — answers a prompt from an embedding-similarity cache before
 * falling through to the model call.
 *
 * Adapted from "GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic
 * Embedding Caching" (arXiv:2411.05276v3). It composes Rivet's existing
 * `embeddingGenerator` and `vectorDatabase` integration contracts (the same I/O
 * contracts used by GetEmbeddingNode / VectorStoreNode / VectorNearestNeighborsNode)
 * to look up a semantically similar cached prompt and return its response when the
 * similarity clears a threshold; otherwise the prompt flows through to the
 * downstream ChatNode and the new answer is written back to the cache.
 *
 * Adaptation note: Rivet's `VectorDatabase.nearestNeighbors` contract returns only
 * the matched data + metadata (the Pinecone implementation discards the similarity
 * score). Rather than depend on a score the contract does not provide, each cache
 * entry stores its own embedding alongside its response, and similarity is computed
 * locally as cosine distance between the query embedding and the returned
 * neighbor's stored embedding. This is a parameter-free proxy for the paper's
 * threshold gate — provider-agnostic and testable without a live vector DB.
 */
export type SemanticCacheNode = ChartNode<'semanticCache', SemanticCacheNodeData>;

export type SemanticCacheNodeData = {
  embeddingIntegration: string;
  useEmbeddingIntegrationInput?: boolean;

  vectorDbIntegration: string;
  useVectorDbIntegrationInput?: boolean;

  collectionId: string;
  useCollectionIdInput?: boolean;

  model?: string;
  useModelInput?: boolean;

  similarityThreshold: number;
  useSimilarityThresholdInput?: boolean;

  cacheResponse: boolean;
};

/** A normalized cache entry recovered from a nearest-neighbor result. */
export type SemanticCacheEntry = {
  response: string;
  embedding: number[];
};

export const DEFAULT_SEMANTIC_CACHE_THRESHOLD = 0.95;

/**
 * Cosine similarity between two equal-length vectors. Returns 0 for empty or
 * mismatched-length inputs, or when either vector is the zero vector.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || a.length !== b.length) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Picks the best cache hit from the recovered entries given a query embedding and
 * a similarity threshold. Pure: no integration calls, so it is unit-testable.
 */
export function selectCacheHit(
  queryEmbedding: number[],
  entries: SemanticCacheEntry[],
  threshold: number,
): { hit: boolean; similarity: number; response: string | null } {
  let bestSimilarity = 0;
  let bestResponse: string | null = null;

  for (const entry of entries) {
    const similarity = cosineSimilarity(queryEmbedding, entry.embedding);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestResponse = entry.response;
    }
  }

  return {
    hit: bestSimilarity >= threshold,
    similarity: bestSimilarity,
    response: bestSimilarity >= threshold ? bestResponse : null,
  };
}

/**
 * Recovers cache entries from a `VectorDatabase.nearestNeighbors` result. Entries
 * are stored as `{ response, embedding }` objects; the Pinecone implementation
 * surfaces them under each match's `metadata`, so we look there first and fall back
 * to `data` for integrations that return the stored object directly.
 */
function recoverEntries(results: ArrayDataValue<ScalarDataValue> | DataValue | undefined): SemanticCacheEntry[] {
  if (results == null) {
    return [];
  }

  const value = (results as { value?: unknown }).value;
  if (!Array.isArray(value)) {
    return [];
  }

  const entries: SemanticCacheEntry[] = [];
  for (const match of value) {
    const source = (match as { metadata?: unknown; data?: unknown }).metadata ?? match;
    const response = (source as { response?: unknown }).response;
    const embedding = (source as { embedding?: unknown }).embedding;
    if (typeof response === 'string' && Array.isArray(embedding)) {
      entries.push({ response, embedding: embedding as number[] });
    }
  }

  return entries;
}

export class SemanticCacheNodeImpl extends NodeImpl<SemanticCacheNode> {
  static create(): SemanticCacheNode {
    return {
      id: newId<NodeId>(),
      type: 'semanticCache',
      title: 'Semantic Cache',
      visualData: { x: 0, y: 0, width: 250 },
      data: {
        embeddingIntegration: 'openai',
        vectorDbIntegration: 'pinecone',
        collectionId: '',
        similarityThreshold: DEFAULT_SEMANTIC_CACHE_THRESHOLD,
        cacheResponse: true,
      },
    };
  }

  getInputDefinitions(): NodeInputDefinition[] {
    const inputs: NodeInputDefinition[] = [
      {
        id: 'prompt' as PortId,
        title: 'Prompt',
        dataType: 'string',
        required: true,
      },
      {
        id: 'response' as PortId,
        title: 'Response',
        dataType: 'string',
        required: false,
      },
    ];

    if (this.data.useEmbeddingIntegrationInput) {
      inputs.push({
        id: 'embeddingIntegration' as PortId,
        title: 'Embedding Integration',
        dataType: 'string',
        required: true,
      });
    }

    if (this.data.useVectorDbIntegrationInput) {
      inputs.push({
        id: 'vectorDbIntegration' as PortId,
        title: 'Vector DB Integration',
        dataType: 'string',
        required: true,
      });
    }

    if (this.data.useCollectionIdInput) {
      inputs.push({
        id: 'collectionId' as PortId,
        title: 'Collection ID',
        dataType: 'string',
        required: true,
      });
    }

    if (this.data.useModelInput) {
      inputs.push({
        id: 'model' as PortId,
        title: 'Embedding Model',
        dataType: 'string',
        required: false,
      });
    }

    if (this.data.useSimilarityThresholdInput) {
      inputs.push({
        id: 'similarityThreshold' as PortId,
        title: 'Similarity Threshold',
        dataType: 'number',
        required: true,
      });
    }

    return inputs;
  }

  getOutputDefinitions(): NodeOutputDefinition[] {
    return [
      {
        id: 'response' as PortId,
        title: 'Response',
        dataType: 'string',
      },
      {
        id: 'hit' as PortId,
        title: 'Cache Hit',
        dataType: 'boolean',
      },
      {
        id: 'similarity' as PortId,
        title: 'Similarity',
        dataType: 'number',
      },
    ];
  }

  getEditors(): EditorDefinition<SemanticCacheNode>[] {
    return [
      {
        type: 'dropdown',
        label: 'Embedding Integration',
        dataKey: 'embeddingIntegration',
        options: [{ label: 'OpenAI', value: 'openai' }],
        useInputToggleDataKey: 'useEmbeddingIntegrationInput',
      },
      {
        type: 'string',
        label: 'Embedding Model',
        dataKey: 'model',
        useInputToggleDataKey: 'useModelInput',
      },
      {
        type: 'dropdown',
        label: 'Vector DB Integration',
        dataKey: 'vectorDbIntegration',
        options: [{ label: 'Pinecone', value: 'pinecone' }],
        useInputToggleDataKey: 'useVectorDbIntegrationInput',
      },
      {
        type: 'string',
        label: 'Collection ID',
        dataKey: 'collectionId',
        useInputToggleDataKey: 'useCollectionIdInput',
      },
      {
        type: 'number',
        label: 'Similarity Threshold',
        dataKey: 'similarityThreshold',
        min: 0,
        max: 1,
        step: 0.01,
        defaultValue: DEFAULT_SEMANTIC_CACHE_THRESHOLD,
        useInputToggleDataKey: 'useSimilarityThresholdInput',
      },
      {
        type: 'toggle',
        label: 'Cache Misses (write new responses back to the cache)',
        dataKey: 'cacheResponse',
      },
    ];
  }

  getBody(): string | undefined {
    return dedent`
      Embedding: ${this.data.useEmbeddingIntegrationInput ? '(input)' : this.data.embeddingIntegration}
      Vector DB: ${this.data.useVectorDbIntegrationInput ? '(input)' : this.data.vectorDbIntegration}
      Threshold: ${this.data.useSimilarityThresholdInput ? '(input)' : this.data.similarityThreshold}
    `;
  }

  static getUIData(): NodeUIData {
    return {
      infoBoxBody: dedent`
        Looks up the prompt in an embedding-similarity cache. If a cached prompt is similar enough (cosine similarity ≥ threshold), its stored response is returned and "Cache Hit" is set to true — skipping the model call entirely.

        On a miss the prompt passes through unchanged so it can reach a ChatNode; when "Cache Misses" is on, the completed response wired into the "Response" input is written back to the cache for future queries.

        Composes the Get Embedding and Vector Store / Vector KNN integrations.
      `,
      infoBoxTitle: 'Semantic Cache Node',
      contextMenuTitle: 'Semantic Cache',
      group: ['AI'],
    };
  }

  async process(inputs: Inputs, context: InternalProcessContext): Promise<Outputs> {
    const prompt = coerceType(inputs['prompt' as PortId], 'string');

    const embeddingIntegration = getInputOrData(this.data, inputs, 'embeddingIntegration');
    const vectorDbIntegration = getInputOrData(this.data, inputs, 'vectorDbIntegration');
    const collectionId = getInputOrData(this.data, inputs, 'collectionId');
    const threshold = getInputOrData(this.data, inputs, 'similarityThreshold', 'number');

    const model = this.data.useModelInput ? coerceType(inputs['model' as PortId], 'string') : this.data.model;

    const embeddingGenerator = getIntegration('embeddingGenerator', embeddingIntegration, context);
    const vectorDb = getIntegration('vectorDatabase', vectorDbIntegration, context);

    const queryEmbedding = await embeddingGenerator.generateEmbedding(prompt, { model });

    const neighbors = await vectorDb.nearestNeighbors(
      { type: 'string', value: collectionId },
      { type: 'vector', value: queryEmbedding } as VectorDataValue,
      1,
    );

    const decision = selectCacheHit(queryEmbedding, recoverEntries(neighbors), threshold);

    if (decision.hit) {
      return {
        ['response' as PortId]: { type: 'string', value: decision.response ?? '' },
        ['hit' as PortId]: { type: 'boolean', value: true },
        ['similarity' as PortId]: { type: 'number', value: decision.similarity },
      };
    }

    // Cache miss: let the prompt fall through to the downstream ChatNode. If a
    // completed response was wired in (e.g. from a ChatNode that already ran),
    // pass it through and optionally persist it for future queries. Note we gate on
    // input presence rather than nullish-coalescing, because coerceTypeOptional
    // coerces an absent string input to "" (not undefined).
    const responseDataValue = inputs['response' as PortId];
    const response = responseDataValue ? coerceTypeOptional(responseDataValue, 'string') : undefined;
    const passthrough = response && response !== '' ? response : prompt;

    if (this.data.cacheResponse && response && response !== '') {
      await vectorDb.store(
        { type: 'string', value: collectionId },
        { type: 'vector', value: queryEmbedding } as VectorDataValue,
        { type: 'object', value: { response, embedding: queryEmbedding } },
        {},
      );
    }

    return {
      ['response' as PortId]: { type: 'string', value: passthrough },
      ['hit' as PortId]: { type: 'boolean', value: false },
      ['similarity' as PortId]: { type: 'number', value: decision.similarity },
    };
  }
}

export const semanticCacheNode = nodeDefinition(SemanticCacheNodeImpl, 'Semantic Cache');
