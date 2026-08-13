import {
  type ChartNode,
  type NodeId,
  type NodeInputDefinition,
  type PortId,
  type NodeOutputDefinition,
} from '../NodeBase.js';
import { nanoid } from 'nanoid/non-secure';
import { NodeImpl, type NodeUIData } from '../NodeImpl.js';
import { dedent } from 'ts-dedent';
import { type EditorDefinition } from '../EditorDefinition.js';
import { type NodeBodySpec } from '../NodeBodySpec.js';
import { nodeDefinition } from '../NodeDefinition.js';
import type { InternalProcessContext } from '../ProcessContext.js';
import type { GptFunction } from '../DataValue.js';
import type { Inputs, Outputs } from '../GraphProcessor.js';
import { coerceTypeOptional, getInputOrData } from '../../utils/index.js';

// Programmatic tool calling (PTC): instead of forcing the model to emit rigid
// JSON tool calls, the tools are exposed as typed code stubs so the model can
// chain and parallelize calls by writing code. Execution happens in a single
// turn inside the sandbox, with each stub dispatched to a backing implementation.
//
// Adapted from "The Bitter Lesson of Tool Calling" (arXiv:2608.06370), which
// exposes tools as typed Python stubs executed in Python. This node reproduces
// that stub-generation + single-turn execution mechanism, reusing Rivet's
// ToolNode tool-definition contract (gpt-function) and CodeNode's sandbox
// (context.codeRunner). The paper's Python execution environment is substituted
// with Rivet's JavaScript sandbox (CodeRunnerOptions), and typed Python stubs
// remain available to mirror the paper's representation for prompting.

export type ProgrammaticToolCallNode = ChartNode<
  'programmaticToolCall',
  ProgrammaticToolCallNodeData
>;

export type ProgrammaticToolCallNodeData = {
  language: 'javascript' | 'python';
  code: string;
  useCodeInput?: boolean;
  allowConsole?: boolean;
};

type JsonSchemaLike = {
  type?: string;
  properties?: Record<string, { type?: string }>;
  required?: string[];
};

type ToolParam = { name: string; schemaType: string | undefined; required: boolean };

const JS_TYPES: Record<string, string> = {
  string: 'string',
  integer: 'number',
  number: 'number',
  boolean: 'boolean',
  array: 'any[]',
  object: 'Record<string, any>',
};

const PY_TYPES: Record<string, string> = {
  string: 'str',
  integer: 'int',
  number: 'float',
  boolean: 'bool',
  array: 'list',
  object: 'dict',
};

function safeName(name: string): string {
  let cleaned = name.trim().replace(/[^a-zA-Z0-9_$]/g, '_');
  if (cleaned.length === 0) {
    cleaned = 'tool';
  }
  if (/^[0-9]/.test(cleaned)) {
    cleaned = `_${cleaned}`;
  }
  return cleaned;
}

function paramsFromSchema(parameters: object | undefined): ToolParam[] {
  const schema = (parameters ?? {}) as JsonSchemaLike;
  const properties = schema.properties ?? {};
  const required = new Set(schema.required ?? []);
  return Object.entries(properties).map(([name, sub]) => ({
    name,
    schemaType: sub?.type,
    required: required.has(name),
  }));
}

function stubJavaScript(tool: GptFunction): string {
  const safe = safeName(tool.name);
  const params = paramsFromSchema(tool.parameters);
  const paramLines = params
    .map((p) => ` * @param {${JS_TYPES[p.schemaType ?? ''] ?? 'any'}} args.${p.name}`)
    .join('\n');
  const description = (tool.description ?? '').trim() || 'No description provided.';
  return [
    '/**',
    ` * ${description}`,
    ' * @param {object} args',
    paramLines,
    ' * @returns {Promise<any>}',
    ' */',
    `async function ${safe}(args) {`,
    '  // dispatched to your implementation at runtime',
    '}',
  ]
    .filter((line) => line !== '')
    .join('\n');
}

function stubPython(tool: GptFunction): string {
  const safe = safeName(tool.name);
  const params = paramsFromSchema(tool.parameters);
  const paramList = params
    .map((p) => `${p.name}: ${PY_TYPES[p.schemaType ?? ''] ?? 'Any'}${p.required ? '' : ' = None'}`)
    .join(', ');
  const description = (tool.description ?? '').trim().replace(/\n/g, ' ') || 'No description provided.';
  return [
    `def ${safe}(${paramList}) -> Any:`,
    `    """${description}"""`,
    '    ...  # dispatched to your implementation at runtime',
  ].join('\n');
}

/** Generates typed code stubs for the given tools, mirroring the paper's tool-as-code representation. */
export function generateToolStubs(tools: GptFunction[], language: 'javascript' | 'python'): string {
  const blocks = tools
    .map((tool) => (language === 'python' ? stubPython(tool) : stubJavaScript(tool)))
    .join('\n\n');
  if (language === 'python') {
    return `from typing import Any\n\n${blocks}`;
  }
  return blocks;
}

/** Builds a system prompt instructing the model to call tools via code instead of JSON. */
export function generateProgrammaticSystemPrompt(
  tools: GptFunction[],
  language: 'javascript' | 'python',
): string {
  const stubs = generateToolStubs(tools, language);
  const parallelHint =
    language === 'python'
      ? 'You may chain calls and parallelize independent ones.'
      : 'You may chain calls and parallelize independent ones with Promise.all.';
  return dedent`
    You are a tool-calling agent. Instead of emitting rigid JSON tool calls, write ${language} that calls the available tool functions. ${parallelHint} The tools are already implemented in the sandbox.

    Available tools:

    ${stubs}

    Write the code that completes the task, then end by returning the final result.
  `;
}

/** Builds an executable script: binds each tool to its implementation, then runs the emitted code. */
export function buildProgrammaticExecutable(tools: GptFunction[], code: string): string {
  const bindings = tools
    .map((tool) => {
      const safe = safeName(tool.name);
      const key = JSON.stringify(safe);
      return `async function ${safe}(args){const f=__impls[${key}];return typeof f==='function'?await f(args):f;}`;
    })
    .join('\n');
  return [
    "const __impls=(inputs['implementations']&&inputs['implementations'].value)||{};",
    bindings,
    'const __ptc_result=await (async () => {',
    code,
    '})();',
    'return { result: { type: "any", value: __ptc_result } };',
  ].join('\n');
}

export class ProgrammaticToolCallNodeImpl extends NodeImpl<ProgrammaticToolCallNode> {
  static create(): ProgrammaticToolCallNode {
    const chartNode: ProgrammaticToolCallNode = {
      type: 'programmaticToolCall',
      title: 'Programmatic Tool Call',
      id: nanoid() as NodeId,
      visualData: {
        x: 0,
        y: 0,
        width: 250,
      },
      data: {
        language: 'javascript',
        code: dedent`
          // Tools are available as async functions. Chain or parallelize freely,
          // then return the final result.
          const weather = await get_weather({ city: 'Seattle' });
          return weather;
        `,
        allowConsole: false,
      },
    };

    return chartNode;
  }

  getInputDefinitions(): NodeInputDefinition[] {
    const inputs: NodeInputDefinition[] = [
      {
        id: 'tools' as PortId,
        title: 'Tools',
        dataType: 'gpt-function[]',
        description: 'Tool definitions (e.g. from Tool nodes) to expose to the model as code stubs.',
      },
      {
        id: 'implementations' as PortId,
        title: 'Implementations',
        dataType: 'object',
        required: false,
        description: 'Map of tool name -> JavaScript function backing each stub, used when executing emitted code.',
      },
    ];

    if (this.data.useCodeInput) {
      inputs.push({
        id: 'code' as PortId,
        title: 'Code',
        dataType: 'string',
        description: 'Code emitted by the model that calls the exposed tools.',
      });
    }

    return inputs;
  }

  getOutputDefinitions(): NodeOutputDefinition[] {
    return [
      {
        id: 'stubs' as PortId,
        title: 'Stubs',
        dataType: 'string',
        description: 'Typed code stubs for the tools, to inject into the model prompt.',
      },
      {
        id: 'systemPrompt' as PortId,
        title: 'System Prompt',
        dataType: 'string',
        description: 'A prompt instructing the model to call the tools via code.',
      },
      {
        id: 'result' as PortId,
        title: 'Result',
        dataType: 'any',
        description: 'The value returned by executing the emitted tool-calling code.',
      },
    ];
  }

  getEditors(): EditorDefinition<ProgrammaticToolCallNode>[] {
    return [
      {
        type: 'dropdown',
        label: 'Stub Language',
        dataKey: 'language',
        options: [
          { value: 'javascript', label: 'JavaScript' },
          { value: 'python', label: 'Python' },
        ],
        defaultValue: 'javascript',
      },
      {
        type: 'toggle',
        label: 'Use Code Input',
        dataKey: 'useCodeInput',
      },
      {
        type: 'code',
        label: 'Code',
        dataKey: 'code',
        language: 'javascript',
        useInputToggleDataKey: 'useCodeInput',
      },
      {
        type: 'toggle',
        label: 'Allow using console',
        dataKey: 'allowConsole',
      },
    ];
  }

  getBody(): string | NodeBodySpec | undefined {
    return `!code_${this.data.language}: programmatic tool calling`;
  }

  static getUIData(): NodeUIData {
    return {
      infoBoxBody: dedent`
        Exposes tools as typed code stubs so the model calls them by writing code
        (chaining and parallelizing naturally) instead of emitting rigid JSON tool calls.
        Generates stubs + a system prompt, and optionally executes the emitted code
        against backing implementations in a single turn.
      `,
      infoBoxTitle: 'Programmatic Tool Call Node',
      contextMenuTitle: 'Programmatic Tool Call',
      group: ['AI'],
    };
  }

  async process(inputs: Inputs, context: InternalProcessContext): Promise<Outputs> {
    const tools = coerceTypeOptional(inputs['tools' as PortId], 'gpt-function[]') ?? [];
    const language: 'javascript' | 'python' = this.data.language === 'python' ? 'python' : 'javascript';

    const outputs: Outputs = {
      ['stubs' as PortId]: { type: 'string', value: generateToolStubs(tools, language) },
      ['systemPrompt' as PortId]: {
        type: 'string',
        value: generateProgrammaticSystemPrompt(tools, language),
      },
    };

    const code = getInputOrData(this.data, inputs, 'code', 'string', 'useCodeInput');
    if (code != null && code.trim() !== '') {
      if (context.codeRunner == null) {
        throw new Error('Programmatic Tool Call node requires a code runner.');
      }
      const executable = buildProgrammaticExecutable(tools, code);
      const execOutputs = await context.codeRunner.runCode(
        executable,
        inputs,
        {
          includeRequire: false,
          includeFetch: false,
          includeRivet: false,
          includeProcess: false,
          includeConsole: this.data.allowConsole ?? false,
        },
      );
      outputs['result' as PortId] = {
        type: 'any',
        value: execOutputs?.['result' as PortId]?.value,
      };
    } else {
      outputs['result' as PortId] = { type: 'control-flow-excluded', value: undefined };
    }

    return outputs;
  }
}

export const programmaticToolCallNode = nodeDefinition(
  ProgrammaticToolCallNodeImpl,
  'Programmatic Tool Call',
);
