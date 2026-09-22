"""Round-4 audit probes. CPU only; temporary fixtures, no production edits."""
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from earthdelta import bank_training as bt, metrics_contract as mc, static_adapter as sa
from earthdelta.bridge import controlled_rollout
from earthdelta.data.make_splits import SplitManifestRow, admit_real_sample
from earthdelta.registry import build_pilot_registry, CandidateRegistry
from earthdelta.contracts import EditPlan
from scripts import s0_gate, r2_fs_bank_train as train_cli, export_upstream_reference as exporter


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    torch.set_num_threads(1)
    results = {}
    rng = np.random.default_rng(92341)
    b, k, h, v, lat, lon = 2, 3, 2, 2, 3, 4
    shape = (b, k, h, v, lat, lon)
    err = rng.normal(size=(b, h, v, lat, lon))
    response = rng.normal(size=shape)
    checks = []
    for qshape in ((h, v, lat, lon), (b, 1, h, v, lat, lon), shape, (lat, 1)):
        q = rng.uniform(.1, 2., size=qshape)
        scale = rng.uniform(.3, 2., size=(h, v, 1, lon))
        weights = np.broadcast_to(q, shape)
        scales = np.broadcast_to(scale, shape)
        manual_gain = np.zeros((b, k))
        manual_loss = np.zeros((b, k))
        manual_diag = np.zeros(shape)
        for bi in range(b):
            for ki in range(k):
                denominator = sum(float(weights[(bi, ki) + idx]) for idx in np.ndindex(h, v, lat, lon))
                for idx in np.ndindex(h, v, lat, lon):
                    pos = (bi, ki) + idx
                    e, u = float(err[(bi,) + idx]), float(response[pos])
                    qe = float(weights[pos]) / float(scales[pos])**2 / denominator
                    manual_diag[pos] = qe
                    manual_gain[bi, ki] += qe * (2*e*u-u*u)
                    manual_loss[bi, ki] += qe * (e-u)**2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        actual = mc.full_objective_gain(t(err), t(response), t(q), scale=t(scale)).numpy()
        loss = mc.full_objective_loss(t(response), t(err), t(q), scale=t(scale)).numpy()
        diag = mc.effective_quadratic_weight(t(q), t(scale), shape).diagonal.numpy()
        assert actual.shape == manual_gain.shape
        for x, y in ((actual, manual_gain), (loss, manual_loss), (diag, manual_diag)):
            np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-13)
        checks.append({'q_shape': qshape, 'gain_max_error': float(abs(actual-manual_gain).max()),
                       'loss_max_error': float(abs(loss-manual_loss).max()),
                       'q_eff_max_error': float(abs(diag-manual_diag).max())})
    try:
        mc.weighted_mse(torch.tensor([1., 2.]), torch.zeros(2), torch.tensor([2., -1.]),
                        convention=mc.WeightConvention.WEIGHTED_MEAN)
    except ValueError as exc:
        negative_rejected = str(exc)
    else:
        raise AssertionError('negative weights accepted')
    assert torch.equal(mc.gain_analytic(t(err), t(response)), mc.gain_calibrated(t(err), t(response)))
    results['metrics'] = {'manual_loop_cases': checks, 'negative_weights_rejected': negative_rejected,
                          'calibration_off_by_default': True}

    complete = {'gate_criteria': {n: True for n in s0_gate.REQUIRED_GATE_CRITERIA},
                'criteria_details': {n: {'passed': True} for n in s0_gate.REQUIRED_GATE_CRITERIA},
                'config_digest': 'audit', 'source_identity': {'modules': {'audit': {}}}}
    variants = {'complete': complete}
    missing = copy.deepcopy(complete); del missing['criteria_details']['outputs_finite']
    variants['never_evaluated'] = missing
    false = copy.deepcopy(complete); false['gate_criteria']['outputs_finite'] = False
    variants['false_criterion'] = false
    for key in ('config_digest', 'source_identity', 'criteria_details'):
        variant = copy.deepcopy(complete); del variant[key]; variants['missing_' + key] = variant
    gate = {}
    for name, value in variants.items():
        try:
            s0_gate._assert_gate_verdict_committable(value)
            gate[name] = 'accepted'
        except s0_gate.GateVerdictNotCommittable as exc:
            gate[name] = str(exc)
    assert gate['complete'] == 'accepted'
    assert all(v != 'accepted' for k, v in gate.items() if k != 'complete')
    results['gate_invariant'] = gate

    before_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            with sa._disable_tf32_for_identity_check():
                inside_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
                raise RuntimeError('audit injected exception')
        except RuntimeError:
            pass
        restored_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        assert inside_flags == (False, False) and restored_flags == (True, True)
        results['tf32_scope'] = {'inside': inside_flags, 'after_exception': restored_flags}
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before_flags

    fixtures = fixture_module('test_bank_training')
    variables = fixtures.variables.__wrapped__()
    bridge = fixtures.fs_bridge.__wrapped__(variables)
    objective = fixtures.objective.__wrapped__(bridge)
    bank = fixtures._bank()
    samples = fixtures._training_samples(bridge)
    state_before = {f'{block}.{name}': p.detach().clone() for block, lora in bank.items() for name, p in lora.named_parameters()}
    cfg = bt.BankTrainConfig(mode='gradient_check', max_updates=1, train_steps=1,
                             hold_steps=1, lead_steps=(1,), num_experts=4,
                             rank_per_expert=2, target_blocks=(0, 1))
    profile = bt.profile_training_step(bridge, bank, samples[0], objective,
                                      expert_index=1, config=cfg, n_measured_steps=1, synthetic=True)
    changes = {f'{block}.{name}': float((p.detach()-state_before[f'{block}.{name}']).abs().max())
               for block, lora in bank.items() for name, p in lora.named_parameters()
               if not torch.equal(p.detach(), state_before[f'{block}.{name}'])}
    assert changes and profile.error is None
    results['profile_mutates_caller_bank'] = changes

    # Hold the real model call while controlled_rollout's edit hooks are active.
    x = samples[0].x_norm
    expected = bridge.forward_validation(x, variables, 6, 1).detach()
    entered, release = threading.Event(), threading.Event()
    def pause_on_controlled(module, args):
        if threading.current_thread().name == 'audit-controlled':
            entered.set()
            if not release.wait(10):
                raise RuntimeError('audit thread timeout')
    handle = bridge.model.register_forward_pre_hook(pause_on_controlled)
    thread_out = {}
    def worker():
        try:
            plan = bt.expert_plan(bt.build_bank_registry(4), 1, hold_steps=1)
            thread_out['output'] = controlled_rollout(bridge, x, variables, 6, 1, plan, bank,
                                                      target_blocks=(0, 1)).detach()
        except Exception as exc:
            thread_out['error'] = repr(exc)
    worker_thread = threading.Thread(target=worker, name='audit-controlled')
    worker_thread.start()
    try:
        assert entered.wait(10)
        contaminated = bridge.forward_validation(x, variables, 6, 1).detach()
    finally:
        release.set(); worker_thread.join(10); handle.remove()
    assert not worker_thread.is_alive() and 'error' not in thread_out, thread_out
    delta = float((contaminated-expected).abs().max())
    assert delta > 0
    results['uncontrolled_forward_during_controlled_rollout'] = {'plain_output_max_change': delta,
        'equals_edited_forward': torch.equal(contaminated, thread_out['output'])}

    with tempfile.TemporaryDirectory(prefix='ed-round4-audit-') as tmp:
        root = Path(tmp)
        times = np.datetime64('2015-03-10T00:00:00') + np.arange(8)*np.timedelta64(6, 'h')
        data = rng.normal(size=(8, len(variables), 16, 32)).astype('float32')
        store = root / '2015.zarr'
        dataset = xr.Dataset({'data': (('time', 'channel', 'lat', 'lon'), data)},
                             coords={'time': times, 'channel': variables,
                                     'lat': np.arange(16), 'lon': np.arange(32)})
        dataset.to_zarr(store, mode='w', consolidated=True)
        issue = int(times[2].astype('datetime64[s]').astype('int64'))
        row = SplitManifestRow(issue_time=issue, valid_time=issue+21600, available_time=issue+43200,
              event_id='audit', split_id='train', normalization_hash='0123456789abcdef',
              grid_hash='grid_identity_that_is_absent_on_disk', availability_source='reanalysis_retrospective')
        admitted = admit_real_sample(row, root, formal=True, data_role='bank_fit', expected_channels=len(variables))
        payload = {'passed': False, 'formal': True, 'results': {'b08_content_verification': False},
                   'content_certificates': [{'passed': False}], 'admission': {'passed': False,
                   'admitted': [admitted.to_dict()]}}
        loaded = sa.load_admitted_samples(payload, bridge, lead_steps=(1,))
        assert len(loaded) == 1
        # Simulate a rewritten store retaining shape/index/spacing but different dates.
        shifted = dataset.assign_coords(time=times + np.timedelta64(24, 'h'))
        shifted.to_zarr(store, mode='w', consolidated=True)
        stale_loaded = sa.load_admitted_samples(payload, bridge, lead_steps=(1,))[0]
        results['admission_boundary'] = {'formal_admitted_without_grid_stamp': admitted.admitted,
            'failed_outer_gate_loaded_samples': len(loaded),
            'mismatched_normalization_hash_accepted': row.normalization_hash != bridge.normalization.digest,
            'stored_issue_time': stale_loaded.issue_time, 'actual_loaded_time': stale_loaded.time_utc,
            'original_time': str(times[2]), 'rewritten_time': str(times[2] + np.timedelta64(24, 'h'))}

        f = fixture_module('test_s0_gate_end_to_end').build_positive_fixture(root / 'gate')
        def fail_commit(value):
            assert value['s0_gate_pass'] is True
            raise RuntimeError('audit failure after provisional PASS')
        with mock.patch.object(s0_gate, '_assert_gate_verdict_committable', side_effect=fail_commit):
            value = s0_gate.run_s0_gate(config=f['config'])
        assert value['s0_gate_pass'] is False and value['verdict_committed'] is False
        results['s0_exception_rollback'] = {k: value[k] for k in ('status', 's0_gate_pass', 'verdict_committed', 'error')}

    calls = []
    def handler(name):
        def invoke(args, ctx):
            calls.append(name)
            return name != 'fs_fit'
        return invoke
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(train_cli, 'build_synthetic_bridge', return_value=(bridge, variables, torch.arange(16))))
        stack.enter_context(mock.patch.object(train_cli, 'build_synthetic_samples', return_value=samples))
        for stage in ('fs_fit', 'fs_freeze', 'bank_train', 'profile', 'horizon_check'):
            stack.enter_context(mock.patch.object(train_cli, 'stage_' + stage, side_effect=handler(stage)))
        rc = train_cli.main(['--synthetic', '--stage', 'all', '--device', 'cpu', '--mode', 'gradient_check',
                             '--max-updates', '1', '--train-steps', '1', '--hold-steps', '1', '--lead-steps', '1'])
    assert rc == 1 and calls == ['fs_fit', 'fs_freeze', 'bank_train', 'profile', 'horizon_check']
    results['failure_continues_dependent_stages'] = {'exit_code': rc, 'called': calls}

    with mock.patch.object(exporter, 'check_xformers_available', return_value=False):
        try:
            exporter.enforce_environment()
        except SystemExit as exc:
            assert exc.code == 2
            results['exporter_missing_xformers_exit_code'] = exc.code
        else:
            raise AssertionError('exporter accepted missing xformers')

    registry = build_pilot_registry(4, metadata={'normalization_hash': 'placeholder_1979_2018'})
    results['registry_placeholder_metadata_accepted'] = registry.validate()
    for bad, name in ((1j, 'complex'), (None, 'empty')):
        try:
            candidate = CandidateRegistry(num_experts=4)
            if bad is not None:
                candidate.register(EditPlan(plan_id='bad', num_experts=4, coefficients=(bad, 0., 0., 0.)))
            candidate.validate()
        except (ValueError, TypeError) as exc:
            results['registry_reject_' + name] = str(exc)
        else:
            raise AssertionError(name + ' registry accepted')
    assert registry.candidate_entries()[0].coefficients[0] == .25
    return results


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        output = run()
    (Path(__file__).parent / 'independent_cpu_probes.json').write_text(json.dumps(output, indent=2) + '\n')
    (Path(__file__).parent / 'independent_cpu_probes.log').write_text(captured.getvalue())
    print(json.dumps(output, indent=2))
