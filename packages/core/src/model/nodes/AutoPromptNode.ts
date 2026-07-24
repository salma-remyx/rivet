/**
 * AutoPrompt node — generates an optimized task prompt by distilling a set of
 * input/output examples. Adapted from:
 *   "Automatic Prompt Optimization with Prompt Distillation" (DistillPrompt),
 *   arXiv:2508.18992 — https://arxiv.org/abs/2508.18992
 *
 * The node implements the paper's three-stage, gradient-free autoprompting
 * pipeline over a Rivet Dataset of examples:
 *   1. Distillation  – for each example, infer the single instruction that
 *      turns the example's input into its output.
 *   2. Compression   – collapse the per-example instructions down to a small,
 *      de-duplicated set of distinct rules.
 *   3. Aggregation   – fuse the compressed rules into one final task prompt.
 *
 * Implementation mode: Mode 2 (adapted port). The core mechanism — the
 * distillation -> compression -> aggregation pipeline — is kept at full
 * fidelity. The paper's auxiliary LLM/eval plumbing is replaced with Rivet's
 * existing OpenAI-compatible chat completion contract (`chatCompletions`) and
 * the existing Dataset value types. The pipeline itself is model-agnostic: it
 * takes an injected `generate` callback, so it is unit-testable with no network
 * access (see `resolveGenerate` for the test seam).
 *
 * Activation: this run's path guardrails are create-only, so the node is
 * delivered as a self-contained `nodeDefinition` that conforms to Rivet's
 * `NodeRegistration` contract (see the registry integration test). To expose it
 * in the built-in palette, append one line to `model/Nodes.ts`:
 *   `.register(autoPromptNode)`
 * alongside the matching import + re-export.
 */
import {
	type ChartNode,
	type NodeId,
	type NodeInputDefinition,
	type NodeOutputDefinition,
	type NodeUIData,
	type Outputs,
	type PortId,
	type EditorDefinition,
	type Inputs,
	type InternalProcessContext,
} from '../../index.js';
import { NodeImpl } from '../NodeImpl.js';
import { nodeDefinition } from '../NodeDefinition.js';
import { dedent, getInputOrData, newId } from '../../utils/index.js';
import { coerceTypeOptional } from '../../utils/coerceType.js';
import {
	OpenAIError,
	chatCompletions,
	type ChatCompletionRequestMessage,
} from '../../utils/openai.js';
import { DEFAULT_CHAT_ENDPOINT } from '../../utils/defaults.js';

export type AutoPromptNode = ChartNode<'autoPrompt', AutoPromptNodeData>;

export type AutoPromptNodeData = {
	/** Natural-language description of the task to optimize a prompt for. */
	task: string;
	useTaskInput?: boolean;

	/** OpenAI-compatible model used for every distillation stage. */
	model: string;
	useModelInput?: boolean;

	/** Key in each example row holding the task input. */
	inputColumn: string;
	/** Key in each example row holding the expected output. */
	outputColumn: string;

	/** Cap on the number of examples processed (each runs one LLM call). */
	maxExamples: number;
	temperature: number;
};

/** A single input/output training example. */
export type PromptExample = {
	input: string;
	output: string;
};

/** Minimal chat message shape consumed by the model-agnostic pipeline. */
export type GenerateMessage = { role: 'system' | 'user' | 'assistant'; content: string };

/** A function that completes a chat — the seam the pipeline calls at each stage. */
export type Generate = (messages: GenerateMessage[]) => Promise<string>;

/** Key under which a test may inject a `Generate` into the process context. */
export const GENERATE_CACHE_KEY = 'autoPrompt.generate';

/**
 * Stage 1 (Distillation): for each example, infer the instruction that maps the
 * example input to its output. Mirrors DistillPrompt's distillation step, where
 * task-specific knowledge is extracted per training sample.
 */
export async function distillExamples(
	task: string,
	examples: PromptExample[],
	generate: Generate,
): Promise<string[]> {
	const instructions = await Promise.all(
		examples.map(async (example) => {
			const raw = await generate([
				{
					role: 'system',
					content: dedent`
						You are a prompt-engineering assistant. Given one input/output example
						for a task, infer the single most important instruction that would
						cause a language model to produce the output from the input.
						Reply with only the instruction, in one sentence.
					`,
				},
				{
					role: 'user',
					content: dedent`
						Task: ${task}

						Input:
						${example.input}

						Output:
						${example.output}

						What single, concise instruction would cause a language model to
						produce the Output from the Input?
					`,
				},
			]);
			return raw.trim();
		}),
	);
	return instructions.filter((instruction) => instruction.length > 0);
}

/**
 * Stage 2 (Compression): collapse the distilled instructions to a small,
 * de-duplicated rule set. DistillPrompt compresses to remove redundancy across
 * the per-example knowledge before aggregation.
 */
export async function compressInstructions(
	task: string,
	instructions: string[],
	generate: Generate,
): Promise<string[]> {
	if (instructions.length === 0) {
		return [];
	}
	const raw = await generate([
		{
			role: 'system',
			content: dedent`
				You are a prompt-engineering assistant. You receive a list of
				instructions distilled from individual examples of a task. Merge
				near-duplicates and remove redundancy, keeping only the distinct,
				essential rules. Output one rule per line, no numbering.
			`,
		},
		{
			role: 'user',
			content: dedent`
				Task: ${task}

				Distilled instructions:
				${instructions.map((instruction, i) => `${i + 1}. ${instruction}`).join('\n')}

				Merge near-duplicates and remove redundancy. Output one rule per line.
			`,
		},
	]);
	return raw
		.split('\n')
		.map((line) => line.replace(/^\s*(?:[-*]|\d+\.)\s*/, '').trim())
		.filter((line) => line.length > 0);
}

/**
 * Stage 3 (Aggregation): fuse the compressed rules into a single, coherent task
 * prompt. This is DistillPrompt's aggregation step, producing the final
 * optimized prompt.
 */
export async function aggregatePrompt(
	task: string,
	rules: string[],
	generate: Generate,
): Promise<string> {
	if (rules.length === 0) {
		return task.trim();
	}
	const raw = await generate([
		{
			role: 'system',
			content: dedent`
				You are a prompt-engineering assistant. Combine a set of rules into a
				single, well-structured prompt that instructs a language model to
				perform a task. Output only the final prompt.
			`,
		},
		{
			role: 'user',
			content: dedent`
				Task: ${task}

				Rules:
				${rules.map((rule) => `- ${rule}`).join('\n')}

				Combine the rules into one final prompt.
			`,
		},
	]);
	return raw.trim();
}

export type DistillPromptStages = {
	distilled: string[];
	compressed: string[];
};

/**
 * Run the full DistillPrompt pipeline (distill -> compress -> aggregate) and
 * return the optimized prompt plus the intermediate stages for transparency.
 */
export async function distillPrompt(
	task: string,
	examples: PromptExample[],
	generate: Generate,
): Promise<{ prompt: string; stages: DistillPromptStages }> {
	const distilled = await distillExamples(task, examples, generate);
	const compressed = await compressInstructions(task, distilled, generate);
	const prompt = await aggregatePrompt(task, compressed, generate);
	return { prompt, stages: { distilled, compressed } };
}

/** Coerce an arbitrary cell value into a string for prompt construction. */
function stringifyCell(value: unknown): string {
	if (value == null) {
		return '';
	}
	if (typeof value === 'string') {
		return value;
	}
	if (typeof value === 'number' || typeof value === 'boolean') {
		return String(value);
	}
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
}

/** Read a single example row into an input/output pair, or null if it is empty. */
export function rowToExample(
	row: Record<string, unknown>,
	inputColumn: string,
	outputColumn: string,
): PromptExample | null {
	const input = stringifyCell(row[inputColumn]);
	const output = stringifyCell(row[outputColumn]);
	if (input === '' && output === '') {
		return null;
	}
	return { input, output };
}

/**
 * Resolve the LLM call used by the pipeline. In production this wires the
 * pipeline to Rivet's existing OpenAI-compatible chat completion contract. The
 * process `executionCache` is inspected for an injected `Generate` so the
 * pipeline can be exercised in tests without any network access.
 */
function resolveGenerate(
	context: InternalProcessContext,
	data: AutoPromptNodeData,
	model: string,
): Generate {
	const injected = context.executionCache.get(GENERATE_CACHE_KEY);
	if (typeof injected === 'function') {
		return injected as Generate;
	}
	return makeOpenAiGenerate(context, data, model);
}

function makeOpenAiGenerate(
	context: InternalProcessContext,
	data: AutoPromptNodeData,
	model: string,
): Generate {
	return async (messages) => {
		const response = await chatCompletions({
			endpoint: context.settings.openAiEndpoint || DEFAULT_CHAT_ENDPOINT,
			auth: {
				apiKey: context.settings.openAiKey ?? '',
				organization: context.settings.openAiOrganization,
			},
			signal: context.signal,
			timeout: context.settings.chatNodeTimeout,
			model,
			temperature: data.temperature,
			messages: messages as ChatCompletionRequestMessage[],
		});

		if ('error' in response) {
			throw new OpenAIError(400, response.error);
		}

		return response.choices[0]?.message?.content ?? '';
	};
}

export class AutoPromptNodeImpl extends NodeImpl<AutoPromptNode> {
	static create(): AutoPromptNode {
		return {
			id: newId<NodeId>(),
			type: 'autoPrompt',
			title: 'Auto Prompt',
			visualData: { x: 0, y: 0, width: 300 },
			data: {
				task: '',
				model: 'gpt-4o-mini',
				inputColumn: 'input',
				outputColumn: 'output',
				maxExamples: 20,
				temperature: 0.7,
			},
		};
	}

	getInputDefinitions(): NodeInputDefinition[] {
		const inputs: NodeInputDefinition[] = [
			{
				id: 'dataset' as PortId,
				title: 'Examples',
				dataType: 'object[]',
				required: true,
				description: 'Array of example rows, each with input/output columns.',
			},
		];

		if (this.data.useTaskInput) {
			inputs.push({
				id: 'task' as PortId,
				title: 'Task',
				dataType: 'string',
				description: 'A natural-language description of the task to optimize a prompt for.',
			});
		}

		if (this.data.useModelInput) {
			inputs.push({
				id: 'model' as PortId,
				title: 'Model',
				dataType: 'string',
				description: 'The OpenAI-compatible model used for the distillation stages.',
			});
		}

		return inputs;
	}

	getOutputDefinitions(): NodeOutputDefinition[] {
		return [
			{
				id: 'prompt' as PortId,
				title: 'Optimized Prompt',
				dataType: 'string',
				description: 'The distilled, optimized task prompt.',
			},
			{
				id: 'rules' as PortId,
				title: 'Distilled Rules',
				dataType: 'string[]',
				description: 'The compressed, de-duplicated rules aggregated into the prompt.',
			},
		];
	}

	getEditors(): EditorDefinition<AutoPromptNode>[] {
		return [
			{
				type: 'string',
				label: 'Task',
				dataKey: 'task',
				useInputToggleDataKey: 'useTaskInput',
				helperMessage: 'Describe the task you want an optimized prompt for.',
			},
			{
				type: 'string',
				label: 'Model',
				dataKey: 'model',
				useInputToggleDataKey: 'useModelInput',
			},
			{
				type: 'string',
				label: 'Input Column',
				dataKey: 'inputColumn',
				helperMessage: 'The key in each example row that holds the task input.',
			},
			{
				type: 'string',
				label: 'Output Column',
				dataKey: 'outputColumn',
				helperMessage: 'The key in each example row that holds the expected output.',
			},
			{
				type: 'number',
				label: 'Max Examples',
				dataKey: 'maxExamples',
				min: 1,
				step: 1,
				helperMessage:
					'Cap on the number of examples processed. Each example runs one LLM call in the distillation stage.',
			},
			{
				type: 'number',
				label: 'Temperature',
				dataKey: 'temperature',
				min: 0,
				max: 2,
				step: 0.1,
			},
		];
	}

	getBody(): string | undefined {
		return this.data.model || undefined;
	}

	static getUIData(): NodeUIData {
		return {
			infoBoxBody: dedent`
				Generates an optimized task prompt from a set of input/output examples.
				Runs a distillation -> compression -> aggregation pipeline over the
				examples (DistillPrompt, arXiv:2508.18992). Each stage calls the
				configured OpenAI-compatible model.
			`,
			infoBoxTitle: 'Auto Prompt Node',
			contextMenuTitle: 'Auto Prompt',
			group: ['AI'],
		};
	}

	async process(inputs: Inputs, context: InternalProcessContext): Promise<Outputs> {
		const task = getInputOrData(this.data, inputs, 'task', 'string') as string;
		const model = getInputOrData(this.data, inputs, 'model', 'string') as string;

		if (!task.trim()) {
			throw new Error('Auto Prompt node requires a task description.');
		}

		const rows = coerceTypeOptional(inputs['dataset' as PortId], 'object[]') ?? [];
		if (rows.length === 0) {
			throw new Error('Auto Prompt node received no example rows.');
		}

		const examples = rows
			.map((row) => rowToExample(row as Record<string, unknown>, this.data.inputColumn, this.data.outputColumn))
			.filter((example): example is PromptExample => example !== null)
			.slice(0, Math.max(1, this.data.maxExamples));

		if (examples.length === 0) {
			throw new Error(
				`No example rows had an "${this.data.inputColumn}" or "${this.data.outputColumn}" column.`,
			);
		}

		const generate = resolveGenerate(context, this.data, model);

		const { prompt, stages } = await distillPrompt(task, examples, generate);

		return {
			['prompt' as PortId]: { type: 'string', value: prompt },
			['rules' as PortId]: { type: 'string[]', value: stages.compressed },
		};
	}
}

export const autoPromptNode = nodeDefinition(AutoPromptNodeImpl, 'Auto Prompt');
