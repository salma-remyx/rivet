import { it, describe } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  globalRivetNodeRegistry,
  ReciprocalRankFusionNodeImpl,
  type NodeConnection,
} from '../../../src/index.js';

const createNode = (data: Partial<ReciprocalRankFusionNodeImpl['chartNode']['data']>) => {
  return new ReciprocalRankFusionNodeImpl({
    ...ReciprocalRankFusionNodeImpl.create(),
    data: {
      ...ReciprocalRankFusionNodeImpl.create().data,
      ...data,
    },
  });
};

const rankedInput = (value: unknown[]) => ({ type: 'any[]' as const, value });

describe('ReciprocalRankFusionNode (integration)', () => {
  it('is registered in the global built-in node registry', () => {
    // Proves the wiring edit in model/Nodes.ts (registerBuiltInNodes chain).
    const types = globalRivetNodeRegistry.getNodeTypes();
    assert.ok(
      types.includes('reciprocalRankFusion'),
      `expected reciprocalRankFusion in registered node types, got: ${types.join(', ')}`,
    );
  });

  it('creates with sensible defaults', () => {
    const node = ReciprocalRankFusionNodeImpl.create();
    assert.strictEqual(node.type, 'reciprocalRankFusion');
    assert.strictEqual(node.data.rrfConstant, 60);
    assert.strictEqual(node.data.topK, 10);
    assert.strictEqual(node.data.queryType, 'balanced');
    assert.strictEqual(node.data.idField, 'id');
  });

  it('exposes dynamic channel inputs based on connections', () => {
    const node = createNode({});
    const connections: NodeConnection[] = [
      { inputNodeId: node.id, inputId: 'channel1' } as NodeConnection,
      { inputNodeId: node.id, inputId: 'channel3' } as NodeConnection,
    ];
    const inputIds = node.getInputDefinitions(connections).map((d) => d.id);
    assert.deepStrictEqual(inputIds, ['channel1', 'channel2', 'channel3']);
  });

  it('fuses ranked lists and ranks the cross-channel consensus first (balanced RRF)', async () => {
    // 'a' is top of channel 1, second in channels 2 and 3 -> consensus winner.
    const node = createNode({ queryType: 'balanced', rrfConstant: 60, topK: 10, idField: 'id' });
    const inputs = {
      channel1: rankedInput([{ id: 'a' }, { id: 'b' }, { id: 'c' }]),
      channel2: rankedInput([{ id: 'b' }, { id: 'a' }, { id: 'd' }]),
      channel3: rankedInput([{ id: 'c' }, { id: 'a' }]),
    };
    const result = await node.process(inputs);

    const ids = (result['results' as never] as { value: { id: string }[] }).value.map((r) => r.id);
    assert.strictEqual(ids[0], 'a', 'consensus document should fuse to rank 1');
    assert.ok(ids.includes('d'), 'document unique to one channel still appears');
    assert.ok(ids.length <= 10);
    // scores port is aligned with results.
    const scores = (result['scores' as never] as { value: number[] }).value;
    assert.strictEqual(scores.length, ids.length);
    // scores are non-increasing (already sorted best-first).
    for (let i = 1; i < scores.length; i++) {
      assert.ok(scores[i - 1]! >= scores[i]!, 'scores must be non-increasing');
    }
  });

  it('adapts fusion weights to query type: sparse query type promotes the sparse channel winner', async () => {
    // channel 1 (dense) and channel 2 (sparse) disagree on the top document.
    const inputs = {
      channel1: rankedInput([{ id: 'p' }, { id: 'q' }]),
      channel2: rankedInput([{ id: 'q' }, { id: 'p' }]),
    };

    // Balanced: equal contribution -> insertion order wins (p first).
    const balanced = createNode({ queryType: 'balanced' });
    const balancedResult = await balanced.process(inputs);
    const balancedIds = (balancedResult['results' as never] as { value: { id: string }[] }).value.map(
      (r) => r.id,
    );
    assert.strictEqual(balancedIds[0], 'p');

    // Sparse query type tilts weight toward channel 2 -> q (sparse's rank-1) wins.
    const sparse = createNode({ queryType: 'sparse' });
    const sparseResult = await sparse.process(inputs);
    const sparseIds = (sparseResult['results' as never] as { value: { id: string }[] }).value.map(
      (r) => r.id,
    );
    assert.strictEqual(sparseIds[0], 'q', 'sparse query type should promote sparse channel winner');
  });

  it('parses custom per-channel weights from the weights input', async () => {
    const inputs = {
      channel1: rankedInput([{ id: 'p' }, { id: 'q' }]),
      channel2: rankedInput([{ id: 'q' }, { id: 'p' }]),
    };
    // Heavy weight on channel 2 via custom weights -> q wins.
    const node = createNode({ queryType: 'custom', weights: '1, 9' });
    const result = await node.process(inputs);
    const ids = (result['results' as never] as { value: { id: string }[] }).value.map((r) => r.id);
    assert.strictEqual(ids[0], 'q');
  });

  it('respects the topK limit', async () => {
    const node = createNode({ topK: 2 });
    const inputs = {
      channel1: rankedInput([{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }]),
      channel2: rankedInput([{ id: 'b' }, { id: 'c' }, { id: 'e' }]),
    };
    const result = await node.process(inputs);
    const ids = (result['results' as never] as { value: { id: string }[] }).value.map((r) => r.id);
    assert.strictEqual(ids.length, 2);
  });

  it('returns empty results when no channels are connected', async () => {
    const node = createNode({});
    const result = await node.process({});
    assert.deepStrictEqual((result['results' as never] as { value: unknown[] }).value, []);
    assert.deepStrictEqual((result['scores' as never] as { value: unknown[] }).value, []);
  });
});
