import assert from 'node:assert/strict';
import { mdpoFinalPageReady, verifyMdpoFinalBrowserPrerequisites } from '../lib/mdpo-final-browser-gate.mjs';

const base = { dataset: { complete: true, sample_count: 33 }, sweep: { trained: 81, evaluated: 81 },
  hyperparameter_selection: { val11_gate_complete: true, evaluated_configuration_count: 81, deployment_eligible: true, selected: { id: 'grid_001' } },
  ablations: { trained: 9, report: { ablations: Array(9).fill('validated') } },
  active_model: { version: 'layout_model_v10_mdpo', sha256: 'a'.repeat(64), mdpo_activated: true },
  historical_reward_model: { sha256: 'b'.repeat(64) }, test11: { status: 'complete_locked_single_test11' } };
const accepted = verifyMdpoFinalBrowserPrerequisites(base);
assert.equal(accepted.outcome, 'activated_locked_test11_complete');
const page = { dataset: '33/33 样本', sweep: '81/81 已评估', ablations: '9/9 已评估', gate: '所选 val11 通过', test: '已完成锁定单次 test11',
  train_metrics: '冻结原始 v10 v10-MDPO · reranker off Δ MDPO − v10', weight_evidence: 'LoRA 更新证据：19 个矩阵' };
assert.equal(mdpoFinalPageReady(page, accepted), true);
assert.equal(mdpoFinalPageReady({ ...page, test: '未锁定' }, accepted), false);
assert.throws(() => verifyMdpoFinalBrowserPrerequisites({ ...base, dataset: { ...base.dataset, sample_count: 18 } }), /requires complete/);
assert.throws(() => verifyMdpoFinalBrowserPrerequisites({ ...base, test11: { status: 'not_locked_before_successful_activation' } }), /requires complete/);
const rejectedInput = { ...base,
  hyperparameter_selection: { ...base.hyperparameter_selection, deployment_eligible: false },
  active_model: { ...base.active_model, version: 'layout_model_v10_anchor_frame_heterogeneous_graph_moe', mdpo_activated: false },
  val11_rejection: { verified: true }, test11: { status: 'not_locked_before_successful_activation' } };
const rejected = verifyMdpoFinalBrowserPrerequisites(rejectedInput);
assert.equal(rejected.outcome, 'all_rejected_original_v10_preserved');
assert.equal(mdpoFinalPageReady({ ...page, gate: '81 个候选全部被拒', test: '未锁定 · 不运行 test11' }, rejected), true);
assert.throws(() => verifyMdpoFinalBrowserPrerequisites({ ...rejectedInput, val11_rejection: { verified: false } }), /requires complete/);
console.log('v10-MDPO final browser prerequisite and rendered status tests passed.');
