import { it, describe } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  CompactChatMessagesNodeImpl,
  TrimChatMessagesNodeImpl,
  globalRivetNodeRegistry,
  type AssistantChatMessage,
  type Inputs,
  type InternalProcessContext,
  type UserChatMessage,
} from '../../../src/index.js';
import { GptTokenizerTokenizer } from '../../../src/integrations/GptTokenizerTokenizer.js';
import type { ChatMessage } from '../../../src/model/DataValue.js';

// The node reads only context.tokenizer; supply the production tokenizer so assertions
// reflect real GPT token counts.
const tokenizer = new GptTokenizerTokenizer();
const context = { tokenizer } as unknown as InternalProcessContext;

const MESSAGE_COUNT = 16;

// Zero-padded markers so no marker is a substring of another (zebra01 != zebra10).
const marker = (i: number) => `zebra${String(i).padStart(2, '0')}`;

const userMessage = (text: string): UserChatMessage => ({ type: 'user', message: text });
const assistantMessage = (text: string): AssistantChatMessage => ({
  type: 'assistant',
  message: text,
  function_call: undefined,
  function_calls: undefined,
});

/** Build an alternating user/assistant conversation where message i contains a unique marker. */
const buildConversation = (count: number): ChatMessage[] => {
  const messages: ChatMessage[] = [];
  for (let i = 0; i < count; i++) {
    const text = `Turn ${i}: remember ${marker(i)} for later, it is important context for the task.`;
    messages.push(i % 2 === 0 ? userMessage(text) : assistantMessage(text));
  }
  return messages;
};

const tokenCount = async (messages: ChatMessage[]): Promise<number> =>
  tokenizer.getTokenCountForMessages(messages, undefined, { node: CompactChatMessagesNodeImpl.create() });

/** Count how many of the original markers (zebra00..zebra{N-1}) survive anywhere in the output. */
const countSurvivingMarkers = (messages: ChatMessage[]): number => {
  const blob = JSON.stringify(messages);
  let surviving = 0;
  for (let i = 0; i < MESSAGE_COUNT; i++) {
    if (blob.includes(marker(i))) {
      surviving++;
    }
  }
  return surviving;
};

const createNode = (overrides: Partial<CompactChatMessagesNodeImpl['data']>) => {
  const base = CompactChatMessagesNodeImpl.create();
  return new CompactChatMessagesNodeImpl({ ...base, data: { ...base.data, ...overrides } });
};

describe('CompactChatMessagesNodeImpl', () => {
  it('can create node', () => {
    const node = CompactChatMessagesNodeImpl.create();
    assert.strictEqual(node.type, 'compactChatMessages');
  });

  // Exercises the wiring edit in model/Nodes.ts (registerBuiltInNodes) via the existing
  // global registry surface — proves the node is reachable as a built-in, not standalone.
  it('is registered as a built-in node', () => {
    assert.ok(globalRivetNodeRegistry.getNodeTypes().includes('compactChatMessages'));
  });

  it('has one input and one output', () => {
    const node = createNode({});
    assert.strictEqual(node.getInputDefinitions().length, 1);
    assert.strictEqual(node.getOutputDefinitions().length, 1);
  });

  it('passes messages through unchanged when already under budget', async () => {
    const node = createNode({ maxTokenCount: 4096 });
    const input = [userMessage('hello'), assistantMessage('hi there')];
    const inputs: Inputs = { input: { type: 'chat-message[]', value: input } };

    const result = await node.process(inputs, context);

    assert.strictEqual(result['compacted'].type, 'chat-message[]');
    assert.deepStrictEqual((result['compacted'].value as ChatMessage[]).length, input.length);
  });

  it('compacts over-budget history within the token budget while preserving recent messages and older information', async () => {
    const input = buildConversation(MESSAGE_COUNT);
    const totalTokens = await tokenCount(input);

    // Budget tight enough to force compaction (~half the conversation).
    const maxTokenCount = Math.ceil(totalTokens / 2);
    const node = createNode({ maxTokenCount });
    const inputs: Inputs = { input: { type: 'chat-message[]', value: input } };

    const result = await node.process(inputs, context);
    const output = result['compacted'].value as ChatMessage[];

    assert.strictEqual(result['compacted'].type, 'chat-message[]');
    assert.ok(output.length >= 1, 'compaction produced output');

    // 1. The result fits the budget.
    const outputTokens = await tokenCount(output);
    assert.ok(outputTokens <= maxTokenCount, `expected <= ${maxTokenCount}, got ${outputTokens}`);

    // 2. Compaction reduced token cost (the economic point of the primitive).
    assert.ok(outputTokens < totalTokens, `expected < ${totalTokens}, got ${outputTokens}`);

    // 3. The most recent message is preserved verbatim (recency fidelity).
    assert.deepStrictEqual(output[output.length - 1], input[input.length - 1]);

    // 4. Older information survives in the consolidated summary. The verbatim recent messages
    //    are exactly the non-system messages in the output; if more markers survive than that,
    //    the extras must come from the densified summary of older turns.
    const verbatimRecent = output.filter((m) => m.type !== 'system').length;
    const survivingMarkers = countSurvivingMarkers(output);
    assert.ok(
      survivingMarkers > verbatimRecent,
      `summary should retain older turns beyond the verbatim recent window (${survivingMarkers} vs ${verbatimRecent})`,
    );
  });

  // Direct comparison to the existing Trim Chat Messages node: for the same token budget,
  // compaction retains strictly more of the original conversation than truncation does.
  // This is the core fidelity claim of the paper's "compacting & consolidation" primitive.
  it('preserves more of the conversation than Trim Chat Messages for the same budget', async () => {
    const input = buildConversation(MESSAGE_COUNT);
    const totalTokens = await tokenCount(input);
    const maxTokenCount = Math.ceil(totalTokens / 2);

    const compactNode = createNode({ maxTokenCount });
    const trimBase = TrimChatMessagesNodeImpl.create();
    const trimNode = new TrimChatMessagesNodeImpl({
      ...trimBase,
      data: { ...trimBase.data, maxTokenCount, removeFromBeginning: true },
    });
    const inputs: Inputs = { input: { type: 'chat-message[]', value: input } };

    const compacted = (await compactNode.process(inputs, context))['compacted'].value as ChatMessage[];
    const trimmed = (await trimNode.process(inputs, context))['trimmed'].value as ChatMessage[];

    const compactSurviving = countSurvivingMarkers(compacted);
    const trimSurviving = countSurvivingMarkers(trimmed);

    assert.ok(
      compactSurviving > trimSurviving,
      `compaction should retain more turns than trim (${compactSurviving} vs ${trimSurviving})`,
    );
  });
});
