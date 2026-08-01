// Compact Chat Messages node — realizes the "compacting & consolidation" primitive
// from "Agentic Context Management" (arXiv:2607.21503v1). Naive context accumulation
// grows token cost quadratically with conversation length; truncation (the existing
// Trim Chat Messages node) buys back budget but discards whole messages, losing
// information. This node keeps the most recent messages verbatim and consolidates
// the older history into a single bounded "conversation so far" summary message, so
// total token cost grows linearly while retaining a densified trace of every older
// message rather than dropping some entirely.
//
// Mode 2 (adapted port): the paper's reference implementation uses an LLM summarizer
// to produce the consolidated summary. That auxiliary is substituted here with a
// deterministic, parameter-free content consolidation so the node is self-contained,
// network-free, and works without API credentials. The core mechanism — keep-recent
// verbatim + collapse older context into one bounded summary message (linear cost,
// better-than-truncation fidelity) — is preserved at full fidelity.
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
import type { TokenizerCallInfo } from '../../integrations/Tokenizer.js';
import { coerceType } from '../../utils/coerceType.js';
import { getInputOrData } from '../../utils/index.js';
import type { ChatMessage, ChatMessageMessagePart, SystemChatMessage } from '../DataValue.js';

export type CompactChatMessagesNodeData = {
  maxTokenCount: number;
  useMaxTokenCountInput?: boolean;

  /**
   * Fraction (0..1) of maxTokenCount reserved for verbatim recent messages.
   * The remaining budget holds the consolidated summary of older messages.
   */
  keepRecentFraction: number;
  useKeepRecentFractionInput?: boolean;
};

export type CompactChatMessagesNode = ChartNode<'compactChatMessages', CompactChatMessagesNodeData>;

export class CompactChatMessagesNodeImpl extends NodeImpl<CompactChatMessagesNode> {
  static create() {
    const chartNode: CompactChatMessagesNode = {
      type: 'compactChatMessages',
      title: 'Compact Chat Messages',
      id: nanoid() as NodeId,
      visualData: {
        x: 0,
        y: 0,
        width: 200,
      },
      data: {
        maxTokenCount: 4096,
        keepRecentFraction: 0.5,
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

    if (this.data.useKeepRecentFractionInput) {
      inputs.push({
        id: 'keepRecentFraction' as PortId,
        title: 'Keep Recent Fraction',
        dataType: 'number',
      });
    }

    return inputs;
  }

  getOutputDefinitions(): NodeOutputDefinition[] {
    return [
      {
        id: 'compacted' as PortId,
        title: 'Compacted',
        dataType: 'chat-message[]',
      },
    ];
  }

  getEditors(): EditorDefinition<CompactChatMessagesNode>[] {
    return [
      {
        type: 'number',
        label: 'Max Token Count',
        dataKey: 'maxTokenCount',
        useInputToggleDataKey: 'useMaxTokenCountInput',
      },
      {
        type: 'number',
        label: 'Keep Recent Fraction',
        dataKey: 'keepRecentFraction',
        useInputToggleDataKey: 'useKeepRecentFractionInput',
      },
    ];
  }

  getBody(): string | NodeBodySpec | undefined {
    return dedent`
      Max Token Count: ${this.data.useMaxTokenCountInput ? '(From Input)' : this.data.maxTokenCount}
      Keep Recent Fraction: ${this.data.useKeepRecentFractionInput ? '(From Input)' : this.data.keepRecentFraction}
    `;
  }

  static getUIData(): NodeUIData {
    return {
      infoBoxBody: dedent`
        Takes an array of chat messages and compacts it to fit a token budget.

        Unlike Trim Chat Messages (which drops messages entirely), this node keeps the most recent messages verbatim and consolidates the older conversation history into a single "conversation so far" summary message. Older messages are preserved in a densified form rather than discarded, so total token cost grows linearly with conversation length while retaining more information than truncation.

        Useful for long-running agentic workflows where conversation history outgrows the context window.
      `,
      infoBoxTitle: 'Compact Chat Messages Node',
      contextMenuTitle: 'Compact Chat Messages',
      group: ['AI'],
    };
  }

  async process(
    inputs: Inputs,
    context: InternalProcessContext<CompactChatMessagesNode>,
  ): Promise<Outputs> {
    const input = coerceType(inputs['input' as PortId], 'chat-message[]');

    const maxTokenCount = getInputOrData(this.data, inputs, 'maxTokenCount', 'number');
    const keepRecentFractionRaw = getInputOrData(this.data, inputs, 'keepRecentFraction', 'number');
    const keepRecentFraction = Math.max(0, Math.min(1, keepRecentFractionRaw));

    const tokenizerInfo: TokenizerCallInfo = {
      node: this.chartNode,
    };

    const totalCount = await context.tokenizer.getTokenCountForMessages(input, undefined, tokenizerInfo);

    // Under budget — nothing to compact, pass through unchanged.
    if (totalCount <= maxTokenCount) {
      return {
        ['compacted' as PortId]: {
          type: 'chat-message[]',
          value: input,
        },
      };
    }

    // Reserve part of the budget for verbatim recent messages; the remainder holds the summary.
    const recentBudget = Math.max(0, Math.floor(maxTokenCount * keepRecentFraction));

    // Greedily take the largest suffix whose token count fits the recent budget.
    let recentSplitIndex = input.length;
    let recentTokens = 0;
    for (let i = input.length - 1; i >= 0; i--) {
      const message = input[i]!;
      const messageTokens = await context.tokenizer.getTokenCountForMessages(
        [message],
        undefined,
        tokenizerInfo,
      );
      if (recentTokens > 0 && recentTokens + messageTokens > recentBudget) {
        break;
      }
      recentTokens += messageTokens;
      recentSplitIndex = i;
    }

    const olderMessages = input.slice(0, recentSplitIndex);
    const recentMessages = input.slice(recentSplitIndex);

    // No older messages to consolidate (recent messages alone already exceed the budget):
    // fall back to trimming from the beginning, matching the behavior of Trim Chat Messages.
    if (olderMessages.length === 0) {
      const trimmed = [...input];
      let tokens = totalCount;
      while (tokens > maxTokenCount && trimmed.length > 0) {
        trimmed.shift();
        tokens = await context.tokenizer.getTokenCountForMessages(trimmed, undefined, tokenizerInfo);
      }
      return {
        ['compacted' as PortId]: {
          type: 'chat-message[]',
          value: trimmed,
        },
      };
    }

    // Bound [summary, ...recent] to the budget, keeping recent messages verbatim wherever
    // possible. fitSummaryToCombined shrinks the summary (recency-weighted: most recent older
    // turns first) to fit alongside the current recent window. If even a minimal summary cannot
    // fit alongside the window, drop the oldest recent message and retry; once the window is
    // empty the summary is truncated to the budget on its own.
    const fullSummary = consolidateMessages(olderMessages);
    const recentWindow = [...recentMessages];
    let compacted: ChatMessage[] = [];
    for (;;) {
      const summaryMessage = await fitSummaryToCombined(
        fullSummary,
        recentWindow,
        maxTokenCount,
        context,
        tokenizerInfo,
      );
      compacted = [summaryMessage, ...recentWindow];
      const compactedTokens = await context.tokenizer.getTokenCountForMessages(
        compacted,
        undefined,
        tokenizerInfo,
      );
      if (compactedTokens <= maxTokenCount || recentWindow.length === 0) {
        break;
      }
      recentWindow.shift();
    }

    return {
      ['compacted' as PortId]: {
        type: 'chat-message[]',
        value: compacted,
      },
    };
  }
}

/** Render a single message part as a compact, token-cheap textual hint. */
function messagePartToText(part: ChatMessageMessagePart): string {
  if (typeof part === 'string') {
    return part;
  }
  switch (part.type) {
    case 'image':
      return '(image)';
    case 'url':
      return `(image: ${part.url})`;
    case 'document':
      return `(document: ${part.title ?? ''})`;
    default:
      return '(attachment)';
  }
}

function messagePartsToText(message: ChatMessageMessagePart | ChatMessageMessagePart[]): string {
  const parts = Array.isArray(message) ? message : [message];
  return parts.map(messagePartToText).join('\n').trim();
}

/**
 * Consolidate a sequence of chat messages into a single bounded system summary message.
 * Each message contributes one densified line (role + gist), so no older message is
 * dropped entirely — its content survives in compressed form.
 */
function consolidateMessages(messages: ChatMessage[]): SystemChatMessage {
  const lines = messages.map((message) => {
    const text = messagePartsToText(message.message);
    switch (message.type) {
      case 'system':
        return `[system] ${text}`;
      case 'user':
        return `user: ${text}`;
      case 'function':
        return `tool ${message.name}: ${text}`;
      case 'assistant': {
        const calls = message.function_calls?.map((call) => call.name).filter(Boolean);
        const called = calls && calls.length > 0 ? ` [called ${calls.join(', ')}]` : '';
        return `assistant: ${text}${called}`;
      }
      default:
        return text;
    }
  });

  return {
    type: 'system',
    message: `Conversation so far (compacted):\n${lines.join('\n')}`,
  };
}

/**
 * Shrink a summary system message so that [summary, ...recent] fits the token budget.
 * Measuring the exact combined array (summary plus the verbatim recent messages) keeps
 * inter-message framing honest. When the body must be truncated, the most recent older
 * turns are kept first (recency-weighted), so the least-relevant history is dropped.
 */
async function fitSummaryToCombined(
  message: SystemChatMessage,
  recent: ChatMessage[],
  maxTokens: number,
  context: InternalProcessContext,
  tokenizerInfo: TokenizerCallInfo,
): Promise<SystemChatMessage> {
  const HEADER = '[Earlier conversation compacted — truncated to fit budget]\n';
  const body = messagePartsToText(message.message);
  const build = (text: string): SystemChatMessage => ({ type: 'system', message: text });
  const measure = (text: string): Promise<number> =>
    context.tokenizer.getTokenCountForMessages([build(text), ...recent], undefined, tokenizerInfo);

  // Fast path: the full body already fits alongside the recent messages.
  if ((await measure(body)) <= maxTokens) {
    return build(body);
  }

  // Binary search the largest body suffix (most recent older turns) such that the header +
  // suffix fits alongside the recent messages.
  let lo = 0;
  let hi = body.length;
  let best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if ((await measure(`${HEADER}${body.slice(body.length - mid)}`)) <= maxTokens) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return build(`${HEADER}${body.slice(body.length - best).trimStart()}`);
}

export const compactChatMessagesNode = nodeDefinition(CompactChatMessagesNodeImpl, 'Compact Chat Messages');
