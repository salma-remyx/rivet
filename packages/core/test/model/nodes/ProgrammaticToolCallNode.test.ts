import { it, describe } from 'node:test';
import { strict as assert } from 'node:assert';
import { dedent } from 'ts-dedent';
import {
  ProgrammaticToolCallNodeImpl,
  IsomorphicCodeRunner,
  type InternalProcessContext,
  type GptFunction,
  type Inputs,
} from '../../../src/index.js';

const weatherTool: GptFunction = {
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string' },
      days: { type: 'integer' },
    },
    required: ['city'],
  },
  strict: false,
};

const toolsInput = (): Inputs => ({
  tools: { type: 'gpt-function[]', value: [weatherTool] },
});

const runnerContext = (): InternalProcessContext =>
  ({ codeRunner: new IsomorphicCodeRunner() }) as unknown as InternalProcessContext;

describe('ProgrammaticToolCallNodeImpl', () => {
  it('can create node', () => {
    const node = ProgrammaticToolCallNodeImpl.create();
    assert.strictEqual(node.type, 'programmaticToolCall');
    assert.strictEqual(node.data.language, 'javascript');
  });

  it('exposes tools, implementations, and (optionally) code inputs', () => {
    const node = new ProgrammaticToolCallNodeImpl(ProgrammaticToolCallNodeImpl.create());
    const ids = node.getInputDefinitions().map((i) => i.id);
    assert.ok(ids.includes('tools'));
    assert.ok(ids.includes('implementations'));
    assert.ok(!ids.includes('code'), 'code input hidden unless useCodeInput is set');

    node.data.useCodeInput = true;
    const idsWithCode = node.getInputDefinitions().map((i) => i.id);
    assert.ok(idsWithCode.includes('code'));
  });

  it('emits stubs, system prompt, and result outputs', () => {
    const node = new ProgrammaticToolCallNodeImpl(ProgrammaticToolCallNodeImpl.create());
    const ids = node.getOutputDefinitions().map((o) => o.id);
    assert.deepStrictEqual(ids, ['stubs', 'systemPrompt', 'result']);
  });

  it('generates typed JavaScript stubs and a code-calling prompt', async () => {
    const node = new ProgrammaticToolCallNodeImpl(ProgrammaticToolCallNodeImpl.create());
    node.data.code = '';
    const result = await node.process(toolsInput(), runnerContext());

    const stubs = result['stubs']!.value as string;
    assert.match(stubs, /async function get_weather\(args\)/);
    assert.match(stubs, /@param \{string\} args\.city/);
    assert.match(stubs, /@param \{number\} args\.days/);

    const prompt = result['systemPrompt']!.value as string;
    assert.match(prompt, /write javascript/i);
    assert.match(prompt, /get_weather/);
    assert.strictEqual(result['result']!.type, 'control-flow-excluded');
  });

  it('can mirror the paper with typed Python stubs', async () => {
    const node = new ProgrammaticToolCallNodeImpl(ProgrammaticToolCallNodeImpl.create());
    node.data.language = 'python';
    node.data.code = '';
    const result = await node.process(toolsInput(), runnerContext());

    const stubs = result['stubs']!.value as string;
    assert.match(stubs, /from typing import Any/);
    assert.match(stubs, /def get_weather\(city: str, days: int = None\) -> Any:/);

    const prompt = result['systemPrompt']!.value as string;
    assert.match(prompt, /write python/i);
  });

  it('executes emitted code against backing implementations in a single turn', async () => {
    const node = new ProgrammaticToolCallNodeImpl(ProgrammaticToolCallNodeImpl.create());
    node.data.code = dedent`
      const weather = await get_weather({ city: 'Seattle' });
      return weather;
    `;

    const inputs: Inputs = {
      ...toolsInput(),
      implementations: {
        type: 'object',
        value: {
          get_weather: (args: { city: string }) => ({ city: args.city, temp: 58 }),
        },
      },
    };

    const result = await node.process(inputs, runnerContext());
    assert.strictEqual(result['result']!.type, 'any');
    assert.deepStrictEqual(result['result']!.value, { city: 'Seattle', temp: 58 });
  });

  it('lets emitted code parallelize independent tool calls', async () => {
    const node = new ProgrammaticToolCallNodeImpl(ProgrammaticToolCallNodeImpl.create());
    node.data.code = dedent`
      const [a, b] = await Promise.all([
        get_weather({ city: 'Seattle' }),
        get_weather({ city: 'Portland' }),
      ]);
      return { a, b };
    `;

    const inputs: Inputs = {
      ...toolsInput(),
      implementations: {
        type: 'object',
        value: {
          get_weather: (args: { city: string }) => ({ city: args.city }),
        },
      },
    };

    const result = await node.process(inputs, runnerContext());
    assert.deepStrictEqual(result['result']!.value, {
      a: { city: 'Seattle' },
      b: { city: 'Portland' },
    });
  });
});
