import {
  type ChartNode,
  type NodeId,
  type NodeInputDefinition,
  type NodeOutputDefinition,
  type PortId,
} from '../NodeBase.js';
import { nanoid } from 'nanoid/non-secure';
import { NodeImpl, type NodeUIData } from '../NodeImpl.js';
import { nodeDefinition } from '../NodeDefinition.js';
import { type Inputs, type Outputs } from '../GraphProcessor.js';
import { type EditorDefinition } from '../../index.js';
import { dedent } from 'ts-dedent';
import { coerceTypeOptional } from '../../utils/coerceType.js';

export type HallucinationScoreNode = ChartNode<'hallucinationScore', HallucinationScoreNodeData>;

export type HallucinationScoreNodeData = {
  // Probability above which `isHallucination` flips true.
  hallucinationThreshold: number;
  // Score band inside which the cheap signal is uncertain and the node
  // should escalate to an expensive verifier downstream.
  escalateLow: number;
  escalateHigh: number;
};

// Compact English stopword set so overlap is not dominated by function words.
const STOPWORDS: ReadonlySet<string> = new Set(
  (
    'a an the and or but if then else of to in on at by for with from as is are was were be been being ' +
    'it its this that these those i you he she we they them his her their our my your has have had do does ' +
    'did not no will would can could should may might must shall so than too very just about into over under'
  ).split(' '),
);

// Below this many answer content tokens the coverage estimate is high-variance;
// the score is blended toward neutral 0.5 until it can be trusted.
const MIN_CONFIDENCE_TOKENS = 6;

function contentTokens(text: string): string[] {
  if (!text) {
    return [];
  }
  const matches = text.toLowerCase().match(/[a-z0-9]+/g);
  if (!matches) {
    return [];
  }
  return matches.filter((token) => token.length >= 2 && !STOPWORDS.has(token));
}

/**
 * Parameter-free proxy for the cost-effective "stage 1" confidence score from
 * "Cost-Effective Hallucination Detection for LLMs": the share of the answer's
 * content tokens that are NOT supported by (lexically entailed in) the source
 * context. Unsupported tokens are content the model produced without grounding
 * in the retrieved source — the core unfaithfulness signal. Returns a
 * hallucination probability in [0, 1], where 0.5 means maximal uncertainty.
 */
function hallucinationProbability(answer: string, context: string): number {
  const answerTokens = contentTokens(answer);
  if (answerTokens.length === 0) {
    return 0.5;
  }

  const contextTokens = contentTokens(context);
  if (contextTokens.length === 0) {
    // No source to ground against: the cheap signal cannot decide, so return
    // neutral and let the node escalate to an expensive verifier downstream.
    return 0.5;
  }

  const contextSet = new Set(contextTokens);
  const supported = answerTokens.reduce((acc, token) => acc + (contextSet.has(token) ? 1 : 0), 0);
  const coverage = supported / answerTokens.length;

  // Short answers give high-variance coverage estimates; blend toward neutral
  // 0.5 until we have enough tokens to trust the estimate.
  const weight = Math.min(1, answerTokens.length / MIN_CONFIDENCE_TOKENS);
  const smoothedCoverage = weight * coverage + (1 - weight) * 0.5;

  return 1 - smoothedCoverage;
}

export class HallucinationScoreNodeImpl extends NodeImpl<HallucinationScoreNode> {
  static create(): HallucinationScoreNode {
    const chartNode: HallucinationScoreNode = {
      type: 'hallucinationScore',
      title: 'Hallucination Score',
      id: nanoid() as NodeId,
      visualData: {
        x: 0,
        y: 0,
        width: 200,
      },
      data: {
        hallucinationThreshold: 0.6,
        escalateLow: 0.35,
        escalateHigh: 0.65,
      },
    };

    return chartNode;
  }

  getInputDefinitions(): NodeInputDefinition[] {
    return [
      {
        dataType: 'string',
        id: 'answer' as PortId,
        title: 'Answer',
      },
      {
        dataType: 'string',
        id: 'context' as PortId,
        title: 'Context',
      },
    ];
  }

  getOutputDefinitions(): NodeOutputDefinition[] {
    return [
      {
        dataType: 'number',
        id: 'score' as PortId,
        title: 'Score',
      },
      {
        dataType: 'boolean',
        id: 'isHallucination' as PortId,
        title: 'Is Hallucination',
      },
      {
        dataType: 'boolean',
        id: 'escalate' as PortId,
        title: 'Escalate',
      },
    ];
  }

  getEditors(): EditorDefinition<HallucinationScoreNode>[] {
    return [
      {
        type: 'number',
        label: 'Hallucination Threshold',
        dataKey: 'hallucinationThreshold',
        defaultValue: 0.6,
        min: 0,
        max: 1,
        step: 0.05,
      },
      {
        type: 'number',
        label: 'Escalate Low',
        dataKey: 'escalateLow',
        defaultValue: 0.35,
        min: 0,
        max: 1,
        step: 0.05,
      },
      {
        type: 'number',
        label: 'Escalate High',
        dataKey: 'escalateHigh',
        defaultValue: 0.65,
        min: 0,
        max: 1,
        step: 0.05,
      },
    ];
  }

  getBody(): string | undefined {
    return 'Score(Answer, Context)';
  }

  static getUIData(): NodeUIData {
    return {
      infoBoxBody: dedent`
        Scores the likelihood that the Answer is a hallucination relative to the source Context, and emits a confidence score plus two boolean gates.

        The score is a parameter-free faithfulness proxy: the share of the answer's content tokens that are NOT supported by the context. This is the cheap "stage 1" signal — wire the Escalate output into an If + Chat node to build a cost-effective cascade that only spends an LLM verifier on answers this cheap score is uncertain about.

        Adapted from "Cost-Effective Hallucination Detection for LLMs".
      `,
      infoBoxTitle: 'Hallucination Score Node',
      contextMenuTitle: 'Hallucination Score',
      group: ['AI'],
    };
  }

  async process(inputs: Inputs): Promise<Outputs> {
    const answer = coerceTypeOptional(inputs['answer' as PortId], 'string') ?? '';
    const context = coerceTypeOptional(inputs['context' as PortId], 'string') ?? '';

    const threshold = this.data.hallucinationThreshold ?? 0.6;
    const escalateLow = this.data.escalateLow ?? 0.35;
    const escalateHigh = this.data.escalateHigh ?? 0.65;

    const score = hallucinationProbability(answer, context);
    const isHallucination = score >= threshold;
    const escalate = score >= escalateLow && score <= escalateHigh;

    return {
      ['score' as PortId]: { type: 'number', value: score },
      ['isHallucination' as PortId]: { type: 'boolean', value: isHallucination },
      ['escalate' as PortId]: { type: 'boolean', value: escalate },
    };
  }
}

export const hallucinationScoreNode = nodeDefinition(HallucinationScoreNodeImpl, 'Hallucination Score');
