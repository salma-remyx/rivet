import { it, describe, mock } from 'node:test';
import { strict as assert } from 'node:assert';
import { ArbitrateChatMessagesNodeImpl, type ChatMessage } from '../../../src/index.js';

// Builds a node with overridden data, mirroring the pattern used by other
// node tests (e.g. ArrayNode.test.ts).
const createNode = (data: Partial<ArbitrateChatMessagesNodeImpl['chartNode']['data']>) => {
  return new ArbitrateChatMessagesNodeImpl({
    ...ArbitrateChatMessagesNodeImpl.create(),
    data: {
      ...ArbitrateChatMessagesNodeImpl.create().data,
      ...data,
    },
  });
};

const run = async (
  node: ArbitrateChatMessagesNodeImpl,
  messages: ChatMessage[],
  context: Record<string, unknown> = {},
) => {
  const inputs = { input: { type: 'chat-message[]' as const, value: messages } };
  const outputs = await node.process(inputs, context as never);
  return outputs['arbitrated' as never] as { type: 'chat-message[]'; value: ChatMessage[] };
};

const textOf = (message: ChatMessage): string => {
  const parts = Array.isArray(message.message) ? message.message : [message.message];
  return parts.map((part) => (typeof part === 'string' ? part : '')).join('\n');
};

describe('ArbitrateChatMessagesNodeImpl', () => {
  it('can create a node of the correct type', () => {
    const node = ArbitrateChatMessagesNodeImpl.create();
    assert.strictEqual(node.type, 'arbitrateChatMessages');
    assert.strictEqual(node.data.maxTokenCount, 2048);
  });

  it('exposes one chat-message[] input and one chat-message[] output', () => {
    const node = createNode({});
    const inputs = node.getInputDefinitions();
    const outputs = node.getOutputDefinitions();
    assert.ok(inputs.some((i) => i.dataType === 'chat-message[]'));
    assert.strictEqual(outputs[0].dataType, 'chat-message[]');
    assert.strictEqual(outputs[0].id, 'arbitrated');
  });

  it('retains the full history in chronological order when the budget is generous', async () => {
    const node = createNode({ maxTokenCount: 1000, ambientSnippetLength: 0 });
    const messages: ChatMessage[] = [
      { type: 'system', message: 'Be concise.' },
      { type: 'user', message: 'Find the key.' },
      {
        type: 'assistant',
        message: 'ok',
        function_call: undefined,
        function_calls: [{ id: '1', name: 'search', arguments: '{}' }],
      },
      { type: 'function', message: 'under the mat', name: 'search' },
      { type: 'assistant', message: 'It is under the mat.' },
    ];

    const out = await run(node, messages);
    assert.strictEqual(out.value.length, 5);
    assert.deepStrictEqual(
      out.value.map((m) => m.type),
      ['system', 'user', 'assistant', 'function', 'assistant'],
    );
    // Nothing compressed at this budget.
    assert.strictEqual(textOf(out.value[0]), 'Be concise.');
  });

  it('drops the lowest-salience messages under a tight budget', async () => {
    // recencyWeight = 1 => only recency matters, so the earliest messages drop first.
    const node = createNode({ maxTokenCount: 6, recencyWeight: 1, ambientSnippetLength: 0 });
    const messages: ChatMessage[] = [
      { type: 'assistant', message: 'earliest note here' },
      { type: 'assistant', message: 'middle note here' },
      { type: 'assistant', message: 'latest' },
    ];

    const out = await run(node, messages);
    // "latest" (5 chars => 2 tokens) and "middle note here" (16 => 4 tokens) fit in 6;
    // "earliest note here" (18 => 5 tokens) does not.
    assert.strictEqual(out.value.length, 2);
    assert.deepStrictEqual(out.value.map(textOf), ['middle note here', 'latest']);
  });

  it('promotes a relevant low-recency item when a focal query is set', async () => {
    // Equal bank weights + low recency weight => relevance drives salience.
    const base = {
      recencyWeight: 0.1,
      ambientSnippetLength: 0,
      directiveBankWeight: 1,
      goalBankWeight: 1,
      actionBankWeight: 1,
      feedbackBankWeight: 1,
      contextBankWeight: 1,
    };
    const messages: ChatMessage[] = [
      { type: 'assistant', message: 'The weather is sunny today.' },
      { type: 'assistant', message: 'Remember to buy milk at the store.' },
      { type: 'assistant', message: 'Plot the quarterly data.' },
      { type: 'assistant', message: 'Done.' },
    ];

    // Budget for exactly two messages (milk = 9 tokens, Done. = 2 tokens => 11).
    const withQuery = createNode({ ...base, maxTokenCount: 12, focalQuery: 'milk' });
    const outWith = await run(withQuery, messages);
    assert.ok(
      outWith.value.some((m) => textOf(m).includes('milk')),
      'relevant milk message should be retained when focal query matches',
    );

    const withoutQuery = createNode({ ...base, maxTokenCount: 12, focalQuery: '' });
    const outWithout = await run(withoutQuery, messages);
    assert.ok(
      !outWithout.value.some((m) => textOf(m).includes('milk')),
      'irrelevant low-recency message should be dropped without a focal query',
    );
  });

  it('compresses a long text message into an ambient snippet under budget', async () => {
    const node = createNode({
      maxTokenCount: 13,
      recencyWeight: 1, // latest message wins outright; long older message is low salience
      ambientSnippetLength: 40,
    });
    const longText = `Alpha ${'x'.repeat(200)}`;
    const messages: ChatMessage[] = [
      { type: 'assistant', message: longText },
      { type: 'assistant', message: 'short' },
    ];

    const out = await run(node, messages);
    assert.strictEqual(out.value.length, 2);
    const ambient = out.value.find((m) => textOf(m).startsWith('Alpha'));
    assert.ok(ambient, 'long message should still be present (compressed)');
    const ambientText = textOf(ambient!);
    assert.ok(ambientText.endsWith('…'), 'ambient message should be truncated with an ellipsis');
    assert.ok(ambientText.length < longText.length, 'ambient message should be shorter than the original');
    assert.strictEqual(textOf(out.value[1]), 'short');
  });

  it('uses the wired tokenizer for budget accounting', async () => {
    // Stub tokenizer reports an enormous token count per message so everything
    // is dropped; the parameter-free fallback would have kept the message.
    const tokenizer = {
      on: () => {},
      getTokenCountForMessages: mock.fn(async () => 9999),
      getTokenCountForString: mock.fn(async () => 9999),
    };
    const node = createNode({ maxTokenCount: 100, ambientSnippetLength: 0 });
    const messages: ChatMessage[] = [{ type: 'assistant', message: 'hi' }];

    const out = await run(node, messages, { tokenizer });
    assert.strictEqual(out.value.length, 0, 'tokenizer-driven budget should drop the oversized message');
    assert.ok(tokenizer.getTokenCountForMessages.mock.callCount() > 0, 'tokenizer must be consulted');
  });

  it('returns an empty array for empty input', async () => {
    const node = createNode({});
    const out = await run(node, []);
    assert.deepStrictEqual(out.value, []);
  });
});
