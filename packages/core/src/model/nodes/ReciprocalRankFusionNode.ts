import {
  type ChartNode,
  type NodeConnection,
  type NodeId,
  type NodeInputDefinition,
  type PortId,
  type NodeOutputDefinition,
} from '../NodeBase.js';
import { NodeImpl, type NodeUIData } from '../NodeImpl.js';
import { nodeDefinition } from '../NodeDefinition.js';
import { nanoid } from 'nanoid/non-secure';
import { type Inputs, type Outputs } from '../GraphProcessor.js';
import { type InternalProcessContext } from '../ProcessContext.js';
import { type EditorDefinition } from '../EditorDefinition.js';
import { dedent } from 'ts-dedent';
import { getInputOrData } from '../../utils/index.js';

/**
 * Reciprocal Rank Fusion node.
 *
 * Fuses the ranked lists produced by several retrieval channels (e.g. a dense
 * Vector KNN node, a sparse/keyword retriever, and a knowledge-graph channel)
 * into a single re-ranked list using query-type-adaptive reciprocal-rank fusion
 * (RRF). With the `balanced` query type this collapses to standard RRF, the
 * fusion baseline APS-RAG measures against; the other query types shift the
 * per-channel weights, mirroring APS-RAG's query-type-adaptive fusion.
 *
 * Adapted from: "A corrective agentic hybrid RAG and an operations-grounded
 * evaluation for a scientific facility" (APS-RAG, arXiv:2607.24663v1).
 *
 * What is ported from the paper (the core mechanism): weighted reciprocal-rank
 * fusion `score(d) = sum_c w_c / (k + rank_c(d))` with the fusion weights
 * adapting to the query type.
 *
 * What is intentionally substituted / scoped out (Mode 2 adaptation):
 *  - The paper's query-type CLASSIFIER is replaced by a parameter-free
 *    `queryType` selector (data supplied by the graph / user) plus a built-in
 *    preset table of query-type -> per-channel weights. The adaptive-weight
 *    mechanism is fully intact; only the upstream classifier is dropped.
 *  - The paper's separate cross-encoder reranker and APS-Bench evaluation
 *    harness are out of scope: the reranker is a distinct downstream APS-RAG
 *    component and evaluation belongs in a downstream PR.
 */
export type ReciprocalRankFusionNode = ChartNode<
  'reciprocalRankFusion',
  ReciprocalRankFusionNodeData
>;

export type ReciprocalRankFusionNodeData = {
  /** RRF constant k (denominator offset). 60 is the standard RRF value. */
  rrfConstant: number;
  useRrfConstantInput?: boolean;

  /** Maximum number of fused results to emit. */
  topK: number;
  useTopKInput?: boolean;

  /**
   * Query type selecting the per-channel weight preset. One of:
   * 'balanced' | 'dense' | 'sparse' | 'kg' | 'custom'.
   * 'balanced' = standard (unweighted) RRF.
   */
  queryType: string;
  useQueryTypeInput?: boolean;

  /**
   * Comma-separated per-channel weights, used only when queryType is 'custom'.
   * Channel i (1-based) takes weights[i-1]; extra channels reuse the last value.
   */
  weights: string;
  useWeightsInput?: boolean;

  /** Field on each ranked item used to identify the same document across channels. */
  idField: string;
};

/**
 * Per-channel weight presets per query type. Channel order is dense, sparse, kg
 * (i.e. connect the dense retriever to channel1, sparse to channel2, etc.).
 * 'balanced' is unweighted RRF — the fusion baseline APS-RAG is measured
 * against; the others tilt the fusion toward a channel as APS-RAG's adaptive
 * weighting does.
 */
const QUERY_TYPE_WEIGHT_PRESETS: Record<string, number[]> = {
  balanced: [1, 1, 1],
  dense: [3, 1, 1],
  sparse: [1, 3, 1],
  kg: [1, 1, 3],
  custom: [1, 1, 1],
};

export class ReciprocalRankFusionNodeImpl extends NodeImpl<ReciprocalRankFusionNode> {
  static create(): ReciprocalRankFusionNode {
    return {
      id: nanoid() as NodeId,
      type: 'reciprocalRankFusion',
      title: 'RRF Fusion',
      visualData: { x: 0, y: 0, width: 250 },
      data: {
        rrfConstant: 60,
        topK: 10,
        queryType: 'balanced',
        weights: '1, 1, 1',
        idField: 'id',
      },
    };
  }

  getInputDefinitions(connections: NodeConnection[]): NodeInputDefinition[] {
    const inputs: NodeInputDefinition[] = [];
    const channelCount = this.#getChannelCount(connections);

    for (let i = 1; i <= channelCount; i++) {
      inputs.push({
        id: `channel${i}` as PortId,
        dataType: 'any[]',
        title: `Channel ${i}`,
        description: dedent`
          A ranked result list from one retrieval channel (dense, sparse, KG, ...).
          Connect channel 1 to your dense retriever, channel 2 to sparse, channel 3 to KG.
        `,
      });
    }

    if (this.data.useQueryTypeInput) {
      inputs.push({
        id: 'queryType' as PortId,
        dataType: 'string',
        title: 'Query Type',
        description: 'One of: balanced, dense, sparse, kg, custom.',
      });
    }

    if (this.data.useWeightsInput) {
      inputs.push({
        id: 'weights' as PortId,
        dataType: 'string',
        title: 'Weights',
        description: 'Comma-separated per-channel weights (used when query type is custom).',
      });
    }

    if (this.data.useRrfConstantInput) {
      inputs.push({
        id: 'rrfConstant' as PortId,
        dataType: 'number',
        title: 'RRF Constant (k)',
      });
    }

    if (this.data.useTopKInput) {
      inputs.push({
        id: 'topK' as PortId,
        dataType: 'number',
        title: 'Top K',
      });
    }

    return inputs;
  }

  getOutputDefinitions(): NodeOutputDefinition[] {
    return [
      {
        id: 'results' as PortId,
        title: 'Fused Results',
        dataType: 'any[]',
        description: 'The fused, re-ranked result list (best first).',
      },
      {
        id: 'scores' as PortId,
        title: 'Scores',
        dataType: 'number[]',
        description: 'The fused RRF score for each emitted result (aligned with results).',
      },
    ];
  }

  getEditors(): EditorDefinition<ReciprocalRankFusionNode>[] {
    return [
      {
        type: 'dropdown',
        label: 'Query Type',
        dataKey: 'queryType',
        options: [
          { label: 'Balanced (standard RRF)', value: 'balanced' },
          { label: 'Dense-favoring', value: 'dense' },
          { label: 'Sparse-favoring', value: 'sparse' },
          { label: 'KG-favoring', value: 'kg' },
          { label: 'Custom', value: 'custom' },
        ],
        useInputToggleDataKey: 'useQueryTypeInput',
      },
      {
        type: 'string',
        label: 'Weights (custom)',
        dataKey: 'weights',
        useInputToggleDataKey: 'useWeightsInput',
      },
      {
        type: 'number',
        label: 'RRF Constant (k)',
        dataKey: 'rrfConstant',
        min: 0,
        step: 1,
        defaultValue: 60,
        useInputToggleDataKey: 'useRrfConstantInput',
      },
      {
        type: 'number',
        label: 'Top K',
        dataKey: 'topK',
        min: 1,
        step: 1,
        defaultValue: 10,
        useInputToggleDataKey: 'useTopKInput',
      },
      {
        type: 'string',
        label: 'ID Field',
        dataKey: 'idField',
      },
    ];
  }

  getBody(): string | undefined {
    return dedent`
      Query Type: ${this.data.useQueryTypeInput ? '(using input)' : this.data.queryType}
      k: ${this.data.useRrfConstantInput ? '(using input)' : this.data.rrfConstant}
      Top K: ${this.data.useTopKInput ? '(using input)' : this.data.topK}
    `;
  }

  static getUIData(): NodeUIData {
    return {
      infoBoxBody: dedent`
        Fuses multiple ranked retrieval channels (dense, sparse, KG, ...) into one
        re-ranked list using query-type-adaptive reciprocal-rank fusion (RRF).

        Wire each retriever's ranked output to a channel input (channel 1 = dense,
        2 = sparse, 3 = KG). Pick a query type to choose the per-channel weights:
        "balanced" is standard RRF; the others tilt the fusion toward a channel.

        Adapted from APS-RAG (arXiv:2607.24663v1).
      `,
      infoBoxTitle: 'RRF Fusion Node',
      contextMenuTitle: 'RRF Fusion',
      group: ['Input/Output'],
    };
  }

  #getChannelCount(connections: NodeConnection[]): number {
    const inputNodeId = this.chartNode.id;
    const channelConnections = connections.filter(
      (connection) => connection.inputNodeId === inputNodeId && connection.inputId.startsWith('channel'),
    );

    let maxChannel = 0;
    for (const connection of channelConnections) {
      const channelNumber = parseInt(connection.inputId.replace('channel', ''));
      if (channelNumber > maxChannel) {
        maxChannel = channelNumber;
      }
    }

    return maxChannel;
  }

  /**
   * Resolve the per-channel weight vector for the given query type and channel
   * count. Channel i (1-based) -> weights[i-1]; extra channels reuse the last
   * preset value; malformed custom weights fall back to balanced.
   */
  #resolveWeights(queryType: string, customWeights: string, channelCount: number): number[] {
    const preset =
      QUERY_TYPE_WEIGHT_PRESETS[queryType] ?? QUERY_TYPE_WEIGHT_PRESETS['balanced']!;

    let base: number[];
    if (queryType === 'custom') {
      const parsed = customWeights
        .split(',')
        .map((part) => Number.parseFloat(part.trim()))
        .filter((weight) => Number.isFinite(weight) && weight >= 0);
      base = parsed.length > 0 ? parsed : QUERY_TYPE_WEIGHT_PRESETS['balanced']!;
    } else {
      base = preset;
    }

    const weights: number[] = [];
    for (let i = 0; i < channelCount; i++) {
      weights.push(base[Math.min(i, base.length - 1)]!);
    }
    return weights;
  }

  async process(inputs: Inputs, _context: InternalProcessContext): Promise<Outputs> {
    const queryType = String(getInputOrData(this.data, inputs, 'queryType')).toLowerCase();
    const customWeights = String(getInputOrData(this.data, inputs, 'weights'));
    const rrfConstant = Number(getInputOrData(this.data, inputs, 'rrfConstant', 'number'));
    const topK = Number(getInputOrData(this.data, inputs, 'topK', 'number'));
    const idField = this.data.idField || 'id';

    const safeK = rrfConstant > 0 ? rrfConstant : 60;
    const safeTopK = topK > 0 ? Math.floor(topK) : 10;

    // Collect each channel's ranked list as an array of items.
    const channels: unknown[][] = [];
    for (let i = 1; ; i++) {
      const portId = `channel${i}` as PortId;
      const input = inputs[portId];
      if (input == null) {
        break;
      }
      const value = input.value;
      const items = Array.isArray(value) ? value : [value];
      channels.push(items);
    }

    if (channels.length === 0) {
      return {
        ['results' as PortId]: { type: 'any[]', value: [] },
        ['scores' as PortId]: { type: 'number[]', value: [] },
      };
    }

    const weights = this.#resolveWeights(queryType, customWeights, channels.length);

    // Weighted reciprocal-rank fusion: score(d) = sum_c w_c / (k + rank_c(d)).
    const scores = new Map<string, { item: unknown; score: number }>();
    for (let c = 0; c < channels.length; c++) {
      const weight = weights[c]!;
      const ranked = channels[c]!;
      for (let rank = 0; rank < ranked.length; rank++) {
        const item = ranked[rank];
        const key = this.#identityKey(item, idField);
        const contribution = weight / (safeK + (rank + 1));
        const existing = scores.get(key);
        if (existing == null) {
          scores.set(key, { item, score: contribution });
        } else {
          existing.score += contribution;
        }
      }
    }

    const fused = [...scores.values()].sort((a, b) => b.score - a.score).slice(0, safeTopK);

    return {
      ['results' as PortId]: {
        type: 'any[]',
        value: fused.map((entry) => entry.item),
      },
      ['scores' as PortId]: {
        type: 'number[]',
        value: fused.map((entry) => entry.score),
      },
    };
  }

  /** Stable identity for a ranked item so the same document is fused across channels. */
  #identityKey(item: unknown, idField: string): string {
    if (item != null && typeof item === 'object') {
      const record = item as Record<string, unknown>;
      if (record[idField] != null) {
        return `${typeof record[idField]}:${String(record[idField])}`;
      }
    }
    try {
      return JSON.stringify(item);
    } catch {
      return String(item);
    }
  }
}

export const reciprocalRankFusionNode = nodeDefinition(ReciprocalRankFusionNodeImpl, 'RRF Fusion');
