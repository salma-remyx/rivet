import { it, describe } from 'node:test';
import { strict as assert } from 'node:assert';
import { globalRivetNodeRegistry, HallucinationScoreNodeImpl } from '../../../src/index.js';

const createNode = () => new HallucinationScoreNodeImpl(HallucinationScoreNodeImpl.create());

const process = async (answer: string, context: string) => {
  const node = createNode();
  return node.process({
    answer: { type: 'string', value: answer },
    context: { type: 'string', value: context },
  });
};

describe('HallucinationScoreNodeImpl', () => {
  it('can create node', () => {
    const node = HallucinationScoreNodeImpl.create();
    assert.strictEqual(node.type, 'hallucinationScore');
  });

  it('has two inputs and three outputs', () => {
    const node = createNode();
    assert.strictEqual(node.getInputDefinitions().length, 2);
    assert.strictEqual(node.getOutputDefinitions().length, 3);
  });

  it('is registered in the built-in node registry', () => {
    // Proves the wiring edit in Nodes.ts: the node is reachable from the
    // registry the Rivet app and executor enumerate at runtime.
    assert.strictEqual(globalRivetNodeRegistry.getDynamicDisplayName('hallucinationScore'), 'Hallucination Score');
  });

  it('processes through the registry end-to-end', async () => {
    const chartNode = globalRivetNodeRegistry.createDynamic('hallucinationScore');
    const impl = globalRivetNodeRegistry.createDynamicImpl(chartNode);
    const result = await impl.process({
      answer: { type: 'string', value: 'The moon is made of green cheese and dragons live there.' },
      context: { type: 'string', value: 'The Moon is Earths only natural satellite.' },
    });
    assert.strictEqual(result['isHallucination'].value, true);
  });

  it('scores a grounded answer as low-risk (not a hallucination, no escalation)', async () => {
    const result = await process(
      'The Eiffel Tower is located in Paris.',
      'The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.',
    );
    const score = result['score'].value as number;
    assert.ok(score <= 0.2, `expected low score, got ${score}`);
    assert.strictEqual(result['isHallucination'].value, false);
    assert.strictEqual(result['escalate'].value, false);
  });

  it('flags an ungrounded answer as a hallucination', async () => {
    const result = await process(
      'The Eiffel Tower is located in Tokyo and costs fifty dollars to ride the dragon.',
      'The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.',
    );
    const score = result['score'].value as number;
    assert.ok(score >= 0.6, `expected high score, got ${score}`);
    assert.strictEqual(result['isHallucination'].value, true);
  });

  it('escalates when the cheap score lands in the uncertain band', async () => {
    const result = await process(
      'The Eiffel Tower is located in Tokyo and costs fifty dollars.',
      'The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.',
    );
    // Score is ~0.57: inside [0.35, 0.65] so the cheap signal cannot decide —
    // the node should escalate to an expensive verifier downstream.
    assert.strictEqual(result['escalate'].value, true);
    assert.strictEqual(result['isHallucination'].value, false);
  });

  it('returns a neutral score and escalates when no source context is provided', async () => {
    const result = await process('Some generated answer with several content tokens here.', '');
    assert.strictEqual(result['score'].value, 0.5);
    assert.strictEqual(result['escalate'].value, true);
  });

  it('returns a neutral score and escalates when the answer is empty', async () => {
    const result = await process('', 'The Moon is Earths only natural satellite.');
    assert.strictEqual(result['score'].value, 0.5);
    assert.strictEqual(result['escalate'].value, true);
  });
});
