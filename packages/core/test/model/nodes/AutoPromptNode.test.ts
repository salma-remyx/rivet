import { it, describe } from 'node:test';
import { strict as assert } from 'node:assert';
// Core (NON-NEW) module: proves the node integrates with Rivet's real registry.
import { NodeRegistration, type InternalProcessContext } from '../../../src/index.js';
// The new capability module under test.
import {
	AutoPromptNodeImpl,
	autoPromptNode,
	aggregatePrompt,
	compressInstructions,
	distillExamples,
	distillPrompt,
	rowToExample,
	GENERATE_CACHE_KEY,
	type Generate,
	type GenerateMessage,
} from '../../../src/model/nodes/AutoPromptNode.js';

/**
 * A deterministic stand-in for an LLM. It routes on the user message so each
 * DistillPrompt stage can be asserted independently without any network calls.
 */
function fakeGenerate(): Generate {
	return async (messages: GenerateMessage[]) => {
		const user = messages.filter((m) => m.role === 'user').at(-1)?.content ?? '';

		if (user.includes('What single, concise instruction')) {
			// Stage 1 — distillation: echo the example's output as the "instruction".
			const output = user.match(/Output:\n([\s\S]*?)\n\n/)?.[1] ?? 'do the task';
			return `Always reply with ${output.trim()}.`;
		}
		if (user.includes('Merge near-duplicates')) {
			// Stage 2 — compression: numbered lines, to exercise number stripping.
			return '1. Rule A\n2. Rule B\n3. Rule A';
		}
		if (user.includes('Combine the rules')) {
			// Stage 3 — aggregation.
			return 'FINAL PROMPT';
		}
		throw new Error(`fakeGenerate received an unexpected prompt:\n${user}`);
	};
}

const EXAMPLES = [
	{ input: 'bonjour', output: 'hello' },
	{ input: 'merci', output: 'thanks' },
];

describe('DistillPrompt pipeline (pure functions)', () => {
	it('distills one instruction per example', async () => {
		const generate = fakeGenerate();
		const distilled = await distillExamples('translate French to English', EXAMPLES, generate);
		assert.strictEqual(distilled.length, EXAMPLES.length);
		assert.ok(distilled.every((instruction) => instruction.length > 0));
	});

	it('compresses instructions and strips numbering/duplicates', async () => {
		const generate = fakeGenerate();
		const compressed = await compressInstructions('task', ['a', 'b', 'c'], generate);
		// Numbers stripped; de-duplication is the model's job, so all lines survive parsing.
		assert.deepStrictEqual(compressed, ['Rule A', 'Rule B', 'Rule A']);
	});

	it('returns the task verbatim when there are no rules to aggregate', async () => {
		const generate = fakeGenerate();
		const prompt = await aggregatePrompt('just the task', [], generate);
		assert.strictEqual(prompt, 'just the task');
	});

	it('runs the full pipeline end to end', async () => {
		const generate = fakeGenerate();
		const { prompt, stages } = await distillPrompt('translate French to English', EXAMPLES, generate);
		assert.strictEqual(stages.distilled.length, EXAMPLES.length);
		assert.deepStrictEqual(stages.compressed, ['Rule A', 'Rule B', 'Rule A']);
		assert.strictEqual(prompt, 'FINAL PROMPT');
	});
});

describe('rowToExample', () => {
	it('reads configured input/output columns', () => {
		assert.deepStrictEqual(rowToExample({ q: '2+2', a: '4' }, 'q', 'a'), { input: '2+2', output: '4' });
	});

	it('stringifies non-string cells', () => {
		assert.deepStrictEqual(rowToExample({ input: 7, output: true }, 'input', 'output'), {
			input: '7',
			output: 'true',
		});
	});

	it('drops rows where both columns are empty', () => {
		assert.strictEqual(rowToExample({ foo: 'x' }, 'input', 'output'), null);
	});
});

describe('AutoPromptNodeImpl', () => {
	it('creates a node with the autoPrompt type', () => {
		const node = AutoPromptNodeImpl.create();
		assert.strictEqual(node.type, 'autoPrompt');
		assert.strictEqual(node.data.inputColumn, 'input');
		assert.strictEqual(node.data.outputColumn, 'output');
	});

	it('declares the example dataset plus optional task/model inputs', () => {
		const node = new AutoPromptNodeImpl({
			...AutoPromptNodeImpl.create(),
			data: { ...AutoPromptNodeImpl.create().data, useTaskInput: true, useModelInput: true },
		});
		const inputIds = node.getInputDefinitions().map((i) => i.id);
		assert.ok(inputIds.includes('dataset'));
		assert.ok(inputIds.includes('task'));
		assert.ok(inputIds.includes('model'));

		const outputIds = node.getOutputDefinitions().map((o) => o.id);
		assert.ok(outputIds.includes('prompt'));
		assert.ok(outputIds.includes('rules'));
	});

	it('runs the distillation pipeline via an injected generator and emits the prompt + rules', async () => {
		const node = new AutoPromptNodeImpl({
			...AutoPromptNodeImpl.create(),
			data: { ...AutoPromptNodeImpl.create().data, task: 'translate French to English' },
		});
		const inputs = {
			dataset: { type: 'object[]' as const, value: EXAMPLES },
		};
		const context = {
			settings: {},
			executionCache: new Map<string, unknown>([[GENERATE_CACHE_KEY, fakeGenerate()]]),
			signal: new AbortController().signal,
		} as unknown as InternalProcessContext;

		const result = await node.process(inputs, context);

		assert.strictEqual(result['prompt'].type, 'string');
		assert.strictEqual(result['prompt'].value, 'FINAL PROMPT');
		assert.deepStrictEqual(result['rules'].value, ['Rule A', 'Rule B', 'Rule A']);
	});

	it('throws when no task is provided', async () => {
		const node = new AutoPromptNodeImpl({ ...AutoPromptNodeImpl.create(), data: { ...AutoPromptNodeImpl.create().data, task: '   ' } });
		const context = { settings: {}, executionCache: new Map(), signal: new AbortController().signal } as unknown as InternalProcessContext;
		await assert.rejects(node.process({ dataset: { type: 'object[]', value: EXAMPLES } }, context));
	});

	it('throws when the dataset has no usable rows', async () => {
		const node = new AutoPromptNodeImpl({
			...AutoPromptNodeImpl.create(),
			data: { ...AutoPromptNodeImpl.create().data, task: 'translate French to English' },
		});
		const context = {
			settings: {},
			executionCache: new Map<string, unknown>([[GENERATE_CACHE_KEY, fakeGenerate()]]),
			signal: new AbortController().signal,
		} as unknown as InternalProcessContext;
		await assert.rejects(
			node.process({ dataset: { type: 'object[]', value: [{ unrelated: 'x' }] } }, context),
			/no example rows/i,
		);
	});
});

describe('AutoPrompt node registry integration', () => {
	it('registers into a real NodeRegistration and is creatable through it', () => {
		// Uses Rivet's actual node registry (from src/index.js) — non-new module.
		const registry = new NodeRegistration();
		const registered = registry.register(autoPromptNode);

		assert.ok(registered.isRegistered('autoPrompt'));
		assert.strictEqual(registered.getDisplayName('autoPrompt'), 'Auto Prompt');

		const node = registered.create('autoPrompt');
		assert.strictEqual(node.type, 'autoPrompt');
	});
});
