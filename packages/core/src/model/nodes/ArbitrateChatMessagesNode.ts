// Function-aware salience arbitration for chat-message[] histories.
//
// Adapted from MemArbiter: Decision-Time Memory Arbitration for Long-Horizon
// LLM Agents (arxiv:2608.02113v1). The paper's core mechanism is ported at
// fidelity: interaction history is decomposed into functional "Memory Banks",
// each item is scored by combining bank-level demand, item-level relevance, and
// a temporal (recency) factor, and a presentation gate decides per item whether
// it is shown fully (focal), compressed (ambient), or dropped, under a unified
// per-step token budget.
//
// Mode-2 substitutions (auxiliary components replaced with target-native,
// parameter-free equivalents so the node stays deterministic and needs no model
// calls of its own):
//   * Atomic items        -> atomic chat messages (Rivet's native unit).
//   * Item-level relevance -> keyword-overlap proxy against a configurable
//                             "focal query" instead of a learned estimator.
//   * Focal/ambient rep.   -> ambient items are truncated to a character snippet
//                             instead of a learned summary.
//   * Temporal gate        -> recency factor derived from message position.
// Output preserves chronological order (only salience/gating is applied) so the
// result stays a valid conversation (no assistant/tool-response reordering).

import {
  type ChartNode,
  type NodeId,
  type PortId,
  type NodeInputDefinition,
  type NodeOutputDefinition,
} from '../../model/NodeBase.js';
import { NodeImpl, type NodeUIData } from '../../model/NodeImpl.js';
import { nanoid } from 'nanoid/non-secure';
import {
  type EditorDefinition,
  type Inputs,
  type InternalProcessContext,
  type NodeBodySpec,
  type Outputs,
} from '../../index.js';
import { dedent } from 'ts-dedent';
import { nodeDefinition } from '../NodeDefinition.js';
import { type ChatMessage } from '../DataValue.js';
import { coerceType } from '../../utils/coerceType.js';
import { getInputOrData } from '../../utils/index.js';

// Parameter-free token estimate used as a fallback when no tokenizer is wired
// (and for sizing ambient snippets). Mirrors the common ~4 chars/token heuristic.
const APPROX_CHARS_PER_TOKEN = 4;

// The five functional Memory Banks. Each chat message is classified into one.
type MemoryBank = 'directive' | 'goal' | 'action' | 'feedback' | 'context';

type Representation = 'focal' | 'ambient' | 'drop';

export type ArbitrateChatMessagesNodeData = {
  maxTokenCount: number;
  useMaxTokenCountInput?: boolean;

  // The current decision focus. Relevance of each item is measured against this
  // string, so it can change every step (decision-time arbitration).
  focalQuery: string;
  useFocalQueryInput?: boolean;

  // Characters retained for an "ambient" (compressed) item. 0 disables
  // compression: items are then only kept (focal) or dropped.
  ambientSnippetLength: number;

  // Blend between relevance and recency in the salience score, in [0, 1].
  // 0 = relevance only, 1 = recency only.
  recencyWeight: number;

  // Bank-level demand weights.
  directiveBankWeight: number;
  goalBankWeight: number;
  actionBankWeight: number;
  feedbackBankWeight: number;
  contextBankWeight: number;
};

export type ArbitrateChatMessagesNode = ChartNode<'arbitrateChatMessages', ArbitrateChatMessagesNodeData>;

/** Concatenate the textual parts of a message (non-text parts contribute nothing). */
function messageText(message: ChatMessage): string {
  const parts = Array.isArray(message.message) ? message.message : [message.message];
  return parts.map((part) => (typeof part === 'string' ? part : '')).join('\n');
}

/** True when every part of the message is plain text (so it can be safely snippeted). */
function isTextOnly(message: ChatMessage): boolean {
  const parts = Array.isArray(message.message) ? message.message : [message.message];
  return parts.every((part) => typeof part === 'string');
}

/** Classify a chat message into one of the five functional Memory Banks. */
function classifyBank(message: ChatMessage): MemoryBank {
  switch (message.type) {
    case 'system':
      return 'directive';
    case 'user':
      return 'goal';
    case 'function':
      return 'feedback';
    case 'assistant':
      return message.function_calls && message.function_calls.length > 0 ? 'action' : 'context';
    default:
      return 'context';
  }
}

/** Keyword-overlap relevance of `text` to `query`, in [0, 1]. Neutral when no query. */
function relevanceScore(text: string, query: string): number {
  const queryTerms = query.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  if (queryTerms.length === 0) {
    return 0.5;
  }
  const textTerms = new Set(text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
  const hits = queryTerms.filter((term) => textTerms.has(term)).length;
  return hits / queryTerms.length;
}

/** Compress a text-only message to a character snippet (ambient representation). */
function toAmbient(message: ChatMessage, snippetLength: number): ChatMessage {
  if (snippetLength <= 0) {
    return message;
  }
  const parts = Array.isArray(message.message) ? message.message : [message.message];
  const compressed = parts.map((part) =>
    typeof part === 'string' && part.length > snippetLength ? `${part.slice(0, snippetLength).trimEnd()}…` : part,
  );
  return { ...message, message: Array.isArray(message.message) ? compressed : compressed[0] };
}

export class ArbitrateChatMessagesNodeImpl extends NodeImpl<ArbitrateChatMessagesNode> {
  static create(): ArbitrateChatMessagesNode {
    const chartNode: ArbitrateChatMessagesNode = {
      type: 'arbitrateChatMessages',
      title: 'Arbitrate Chat Messages',
      id: nanoid() as NodeId,
      visualData: {
        x: 0,
        y: 0,
        width: 250,
      },
      data: {
        maxTokenCount: 2048,
        focalQuery: '',
        ambientSnippetLength: 200,
        recencyWeight: 0.5,
        directiveBankWeight: 1.2,
        goalBankWeight: 1.3,
        actionBankWeight: 0.9,
        feedbackBankWeight: 0.8,
        contextBankWeight: 0.6,
      },
    };

    return chartNode;
  }

  getInputDefinitions(): NodeInputDefinition[] {
    const inputs: NodeInputDefinition[] = [
      {
        id: 'input' as PortId,
        title: 'Input',
        dataType: 'chat-message[]',
      },
    ];

    if (this.data.useMaxTokenCountInput) {
      inputs.push({
        id: 'maxTokenCount' as PortId,
        title: 'Max Token Count',
        dataType: 'number',
      });
    }

    if (this.data.useFocalQueryInput) {
      inputs.push({
        id: 'focalQuery' as PortId,
        title: 'Focal Query',
        dataType: 'string',
      });
    }

    return inputs;
  }

  getOutputDefinitions(): NodeOutputDefinition[] {
    return [
      {
        id: 'arbitrated' as PortId,
        title: 'Arbitrated',
        dataType: 'chat-message[]',
      },
    ];
  }

  getEditors(): EditorDefinition<ArbitrateChatMessagesNode>[] {
    return [
      {
        type: 'number',
        label: 'Max Token Count',
        dataKey: 'maxTokenCount',
        useInputToggleDataKey: 'useMaxTokenCountInput',
      },
      {
        type: 'string',
        label: 'Focal Query',
        dataKey: 'focalQuery',
        useInputToggleDataKey: 'useFocalQueryInput',
      },
      {
        type: 'number',
        label: 'Ambient Snippet Length (chars)',
        dataKey: 'ambientSnippetLength',
        min: 0,
      },
      {
        type: 'number',
        label: 'Recency Weight',
        dataKey: 'recencyWeight',
        min: 0,
        max: 1,
        step: 0.1,
      },
      {
        type: 'number',
        label: 'Directive Bank Weight',
        dataKey: 'directiveBankWeight',
        min: 0,
        step: 0.1,
      },
      {
        type: 'number',
        label: 'Goal Bank Weight',
        dataKey: 'goalBankWeight',
        min: 0,
        step: 0.1,
      },
      {
        type: 'number',
        label: 'Action Bank Weight',
        dataKey: 'actionBankWeight',
        min: 0,
        step: 0.1,
      },
      {
        type: 'number',
        label: 'Feedback Bank Weight',
        dataKey: 'feedbackBankWeight',
        min: 0,
        step: 0.1,
      },
      {
        type: 'number',
        label: 'Context Bank Weight',
        dataKey: 'contextBankWeight',
        min: 0,
        step: 0.1,
      },
    ];
  }

  getBody(): string | NodeBodySpec | undefined {
    return dedent`
      Max Tokens: ${this.data.useMaxTokenCountInput ? '(From Input)' : this.data.maxTokenCount}
      Focal Query: ${this.data.useFocalQueryInput ? '(From Input)' : this.data.focalQuery || '(none)'}
      Recency Weight: ${this.data.recencyWeight}
      Banks: directive ${this.data.directiveBankWeight} · goal ${this.data.goalBankWeight} · action ${this.data.actionBankWeight} · feedback ${this.data.feedbackBankWeight} · context ${this.data.contextBankWeight}
    `;
  }

  static getUIData(): NodeUIData {
    return {
      infoBoxBody: dedent`
        Function-aware salience arbitration for a chat-message[] history. Each message is sorted into one of five functional Memory Banks (directive / goal / action / feedback / context), scored from bank demand + relevance to a focal query + recency, and presented fully (focal), compressed to a snippet (ambient), or dropped — under a unified per-step token budget. Output stays in chronological order.

        Use between retrieval/memory and a prompt node so the most decision-relevant context — not just the most recent — guides the next step.
      `,
      infoBoxTitle: 'Arbitrate Chat Messages Node',
      contextMenuTitle: 'Arbitrate Chat Messages',
      group: ['AI'],
    };
  }

  async process(inputs: Inputs, context: InternalProcessContext<ArbitrateChatMessagesNode>): Promise<Outputs> {
    const inputValue = inputs['input' as PortId];
    const messages: ChatMessage[] = inputValue ? coerceType(inputValue, 'chat-message[]') : [];

    const maxTokenCount = getInputOrData(this.data, inputs, 'maxTokenCount', 'number');
    const focalQuery = getInputOrData(this.data, inputs, 'focalQuery', 'string');
    const ambientSnippetLength = this.data.ambientSnippetLength;
    const recencyWeight = clamp(this.data.recencyWeight, 0, 1);

    const bankWeights: Record<MemoryBank, number> = {
      directive: this.data.directiveBankWeight,
      goal: this.data.goalBankWeight,
      action: this.data.actionBankWeight,
      feedback: this.data.feedbackBankWeight,
      context: this.data.contextBankWeight,
    };

    if (messages.length === 0 || maxTokenCount <= 0) {
      return {
        ['arbitrated' as PortId]: { type: 'chat-message[]', value: [] },
      };
    }

    const countTokens = async (msgs: ChatMessage[]): Promise<number> => {
      if (context?.tokenizer) {
        return context.tokenizer.getTokenCountForMessages(msgs, undefined, {
          node: this.chartNode,
        });
      }
      return msgs.reduce((sum, msg) => sum + Math.ceil(messageText(msg).length / APPROX_CHARS_PER_TOKEN), 0);
    };

    const n = messages.length;

    // Score every item by bank demand * (relevance/recency blend).
    const scored = messages.map((message, index) => {
      const bank = classifyBank(message);
      const relevance = relevanceScore(messageText(message), focalQuery);
      const recency = n > 1 ? (index + 1) / n : 1;
      const score = bankWeights[bank] * ((1 - recencyWeight) * relevance + recencyWeight * recency);
      return { message, index, score, recency };
    });

    // Allocate the budget in salience order (highest score first).
    const priority = [...scored].sort((a, b) => b.score - a.score || b.recency - a.recency);

    const decisions: Representation[] = new Array(n).fill('drop');
    let remaining = maxTokenCount;

    for (const item of priority) {
      if (remaining <= 0) {
        break;
      }
      const fullTokens = await countTokens([item.message]);
      if (fullTokens <= remaining) {
        decisions[item.index] = 'focal';
        remaining -= fullTokens;
      } else if (ambientSnippetLength > 0 && isTextOnly(item.message)) {
        const snippetTokens = Math.ceil(ambientSnippetLength / APPROX_CHARS_PER_TOKEN);
        if (snippetTokens <= remaining) {
          decisions[item.index] = 'ambient';
          remaining -= snippetTokens;
        }
      }
    }

    // Re-emit in chronological order, applying each representation decision.
    const arbitrated: ChatMessage[] = [];
    for (const item of scored) {
      const decision = decisions[item.index];
      if (decision === 'focal') {
        arbitrated.push(item.message);
      } else if (decision === 'ambient') {
        arbitrated.push(toAmbient(item.message, ambientSnippetLength));
      }
    }

    return {
      ['arbitrated' as PortId]: {
        type: 'chat-message[]',
        value: arbitrated,
      },
    };
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export const arbitrateChatMessagesNode = nodeDefinition(ArbitrateChatMessagesNodeImpl, 'Arbitrate Chat Messages');
