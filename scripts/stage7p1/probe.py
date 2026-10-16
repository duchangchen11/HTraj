"""Bounded real-data admission; no epochs, optimizer updates, or loss changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import copy
import json
import subprocess
import traceback
import numpy as np
import torch
from src.agentformer_admission.causal import (build_context, labels_for_context,
    original_training_packet, inference_packet, UnlabeledContextError)
from src.data.preprocessing import Recording, sha256
from src.socialvae_causal.data import array_sha
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new, state_info
from scripts.stage7p1.runtime import load_model, predict, ResourceMeter

OUT = Path('results/tables/stage7p1')
CACHE = Path('experiments/stage7p1/cache')


def record(name):
    scene = 'HOTEL' if name == 'biwi_hotel' else ('ZARA2' if name == 'crowds_zara02' else 'UNIV')
    return Recording.read(Path('data/raw/eth_ucy')/(name+'.txt'), scene)


def select_cases(audit):
    cases = copy.deepcopy(audit['cases'])
    rows = read_csv(OUT/'context_windows.csv.gz')
    for rec in ['biwi_hotel', 'crowds_zara02']:
        for n in [1, 2, 3]:
            # Protocol addendum frozen before model execution: fill unavailable
            # validation coverage from original source TRAIN regions only.
            if n == 3 and any(c['recording'] == rec and c['people'] >= 3 and not c['incomplete_future_nodes'] for c in cases):
                continue
            if n < 3 and any(c['recording'] == rec and c['people'] == n and not c['incomplete_future_nodes'] for c in cases):
                continue
            match = next((r for r in rows if r['recording'] == rec and r['split'] == 'train'
                and int(r['people']) == n and int(r['incomplete_future_nodes']) == 0 and int(r['targets']) > 0), None)
            if match is None:
                continue
            c = dict(match)
            for key in ['people', 'start_frame', 'targets', 'partial_history_excluded', 'incomplete_future_nodes', 'no_future_nodes']:
                c[key] = int(c[key])
            c['reasons'] = [f'supplement_original_train_region_N{n}']
            cases.append(c)
        assert sum(c['recording'] == rec for c in cases) <= 6
    return cases


def real_batches(model, build, cases):
    results = []
    for case_index, c in enumerate(cases):
        r = record(c['recording'])
        context = build_context(r, c['start_frame'])
        row = dict(**c, optimizer_updates=0, formal_training=False)
        print('Real case', r.name, context.start, 'N', len(context.ids), flush=True)
        try:
            with ResourceMeter() as meter:
                prediction, fingerprints = predict(model, context)
                row['inference_finite'] = bool(np.isfinite(prediction).all())
                row['input_fingerprints'] = fingerprints
                row['prediction_sha256'] = array_sha(prediction)
                try:
                    packet = original_training_packet(context, r)
                except UnlabeledContextError as error:
                    row.update(original_loss_status='BLOCKED_UNLABELED_CONTEXT', reason=str(error),
                        true_supervised_loss=None, gradients_finite=None)
                else:
                    model.train()
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(17)
                    torch.cuda.manual_seed_all(17)
                    model.set_data(packet)
                    model.forward()
                    loss, terms, _ = model.compute_loss()
                    assert torch.isfinite(loss), 'Nonfinite true original loss'
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    assert grads and all(torch.isfinite(g).all() for g in grads)
                    grad_norm = sum(float(g.detach().double().square().sum()) for g in grads)**.5
                    assert grad_norm > 0
                    row.update(original_loss_status='PASS_ALL_CONTEXT_LABELS_COMPLETE',
                        true_supervised_loss=float(loss.detach()), original_loss_terms=terms,
                        gradient_L2=grad_norm, gradients_finite=True,
                        gradient_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None))
                    del loss
                    model.zero_grad(set_to_none=True)
                # Strict inference-state restoration is meaningful even when
                # the original training objective is not defined for this case.
                path = CACHE/f'real_case_{case_index}_random_state.pth'
                with path.open('xb') as f:
                    torch.save(dict(model_dict=model.state_dict(), optimizer_updates=0), f)
                restored = build().eval()
                restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_dict'], strict=True)
                b, _ = predict(restored, context)
                assert np.array_equal(prediction, b)
                row.update(strict_checkpoint_restore=True, restored_prediction_max_difference=float(np.abs(prediction-b).max()),
                    checkpoint=str(path), checkpoint_sha256=sha256(path))
                del restored
            row['resource'] = meter.result
        except Exception:
            row['unexpected_failure'] = traceback.format_exc()
            if 'meter' in locals() and hasattr(meter, 'result'):
                row['resource'] = meter.result
        results.append(row)
        json_new(OUT/f'real_cases/case_{case_index}.json', row)
    return results


def counterfactuals(model, audit):
    from scripts.stage7p1.raw_variants import variants
    results = []
    for name in ['biwi_hotel', 'crowds_zara02']:
        case = next(c for c in audit['cases'] if c['recording'] == name and 'mixed_future' in c['reasons'])
        start = case['start_frame']
        original = record(name)
        c = build_context(original, start)
        y, valid = labels_for_context(c, original)
        target = int(c.ids[valid.all(1)][0])
        inputs = np.loadtxt(Path('data/raw/eth_ucy')/(name+'.txt'))
        raw = inputs[(inputs[:, 0] >= start) & (inputs[:, 0] <= start+190)]
        predictions, features = {}, {}
        history_sha = array_sha(c.history)
        with torch.no_grad():
            for k in [1, 20]:
                predictions[k], features[k] = predict(model, c, k)
        for variant_name, modified, details in variants(raw, c, target):
            folder = CACHE/'raw_counterfactuals'/name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/(variant_name+'.txt')
            with path.open('x') as f:
                np.savetxt(f, modified, fmt=['%d', '%d', '%.9g', '%.9g'])
            mutated = Recording.read(path, original.scene)
            mutated.name = name
            rebuilt = build_context(mutated, start)
            hist_before = raw[raw[:, 0] <= start+70]
            hist_after = modified[modified[:, 0] <= start+70]
            row = dict(recording=name, start_frame=start, variant=variant_name, anchor_target_id=target,
                raw_path=str(path), raw_sha256=sha256(path), details=details,
                history_rows_exact=bool(np.array_equal(hist_before, hist_after)),
                context_ids_equal=bool(np.array_equal(c.ids, rebuilt.ids)),
                partial_ids_equal=bool(np.array_equal(c.partial_ids, rebuilt.partial_ids)),
                history_sha_equal=history_sha == array_sha(rebuilt.history))
            for k in [1, 20]:
                prediction, fingerprints = predict(model, rebuilt, k)
                difference = float(np.abs(prediction-predictions[k]).max())
                row[f'K{k}_prediction_max_difference'] = difference
                row[f'K{k}_historical_model_fingerprints_equal'] = fingerprints == features[k]
                row[f'K{k}_prediction_equal'] = bool(np.array_equal(prediction, predictions[k]))
            row['passed'] = all(row[key] for key in ['history_rows_exact', 'context_ids_equal', 'partial_ids_equal', 'history_sha_equal',
                'K1_prediction_equal', 'K20_prediction_equal', 'K1_historical_model_fingerprints_equal', 'K20_historical_model_fingerprints_equal'])
            assert details['changed_future_rows'] > 0
            results.append(row)
            json_new(OUT/f'counterfactuals/{name}_{variant_name}.json', row)
    return results


def negative_posterior_probe(model, audit):
    results = []
    for name in ['biwi_hotel', 'crowds_zara02']:
        case = next(c for c in audit['cases'] if c['recording'] == name and 'mixed_future' in c['reasons'])
        r = record(name)
        c = build_context(r, case['start_frame'])
        y, valid = labels_for_context(c, r)
        supervised = valid.all(1)
        rows, means = [], []
        for offset in [0., 100., None]:
            packet = inference_packet(c)
            for i in range(len(c.ids)):
                fill = np.full((12, 2), np.nan, np.float32) if offset is None else np.tile(c.history[i, -1]+offset, (12, 1))
                fill[valid[i]] = y[i, valid[i]]
                packet['fut_motion_3D'][i] = torch.tensor(fill, requires_grad=True)
                # Even eliminating ALL unsupervised agents' reconstruction
                # labels cannot eliminate their future-token/KL dependency.
                packet['fut_motion_mask'][i] = torch.full((12,), float(supervised[i]))
            model.eval()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            model.set_data(packet)
            model.forward()
            loss, terms, _ = model.compute_loss()
            q = model.data['q_z_dist'].mu
            means.append(q[supervised].detach().cpu().numpy())
            grad = torch.autograd.grad(q[supervised].sum(), packet['fut_motion_3D'], allow_unused=True)
            dependencies = [g[~torch.tensor(valid[i])].abs().sum().item()
                for i, g in enumerate(grad) if not supervised[i] and g is not None]
            rows.append(dict(placeholder='NaN' if offset is None else f'last_position_plus_{offset}m',
                loss_finite=bool(torch.isfinite(loss)), loss=float(loss.detach()) if torch.isfinite(loss) else None,
                loss_terms={k: v if np.isfinite(v) else None for k, v in terms.items()},
                supervised_posterior_finite=bool(torch.isfinite(q[supervised]).all()),
                supervised_q_gradient_wrt_missing_neighbor_future_L1=sum(dependencies) if offset is not None else None))
            del loss, q
        delta = float(np.abs(means[0]-means[1]).max())
        row = dict(recording=name, start_frame=c.start, people=len(c.ids), supervised_targets=int(supervised.sum()),
            incomplete_context_ids=c.ids[~supervised].tolist(), completions=rows,
            supervised_posterior_mu_max_difference=delta,
            mask_only_solution_refuted=bool(delta > 1e-6 and rows[0]['supervised_q_gradient_wrt_missing_neighbor_future_L1'] > 0 and not rows[2]['loss_finite']),
            interpretation='INVALID placeholder diagnostic, never training supervision or performance; no optimizer updates',
            optimizer_updates=0)
        json_new(OUT/f'negative_probes/{name}.json', row)
        results.append(row)
    model.zero_grad(set_to_none=True)
    return results


def resource_probe(model, audit):
    case = audit['maximum_source_context']
    r = record(case['recording'])
    c = build_context(r, case['start_frame'])
    model.data = None
    model.zero_grad(set_to_none=True)
    row = dict(**case, tested_mode='learned-prior K20 inference ONLY',
        full_original_training_status='BLOCKED_UNLABELED_CONTEXT', prior_stage7_N=57,
        context_truncated=False, training_memory_feasibility='NOT_ESTABLISHED', optimizer_updates=0)
    try:
        with ResourceMeter() as meter:
            prediction, _ = predict(model, c, 20)
            row['prediction_finite'] = bool(np.isfinite(prediction).all())
        row['resource'] = meter.result
    except Exception:
        row['failure'] = traceback.format_exc()
        row['resource'] = meter.result
    json_new(OUT/'maximum_context_resource.json', row)
    return row


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    audit = json.load(open(OUT/'data_audit.json'))
    cases = select_cases(audit)
    model, build, meta, settings = load_model()
    json_new(OUT/'execution_setup.json', dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        source_SHA256={str(p): sha256(p) for p in [*Path('scripts/stage7p1').glob('*.py'), *Path('src/agentformer_admission').glob('*.py')]},
        official_commit=meta['commit'], model_settings=settings, model_state=state_info(model),
        cases=cases, torch=torch.__version__, numpy=np.__version__,
        GPU=torch.cuda.get_device_name(), torch_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        dependency_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        compatibility='Existing Stage7 runtime _LinearWithBias alias only; no official source changed; original CLI/DataGenerator NOT causal-admitted',
        optimizer_updates=0, formal_epochs=0, ETH_evaluated=False))
    cf = counterfactuals(model, audit)
    print('Raw-data counterfactuals', sum(r['passed'] for r in cf), '/', len(cf), flush=True)
    real = real_batches(model, build, cases)
    negative = negative_posterior_probe(model, audit)
    resource = resource_probe(model, audit)
    json_new(OUT/'execution_summary.json', dict(raw_counterfactuals_passed=all(r['passed'] for r in cf),
        raw_counterfactual_count=len(cf), real_cases=len(real),
        true_loss_backward_passes=sum(r.get('original_loss_status') == 'PASS_ALL_CONTEXT_LABELS_COMPLETE' for r in real),
        mixed_label_cases_blocked=sum(r.get('original_loss_status') == 'BLOCKED_UNLABELED_CONTEXT' for r in real),
        unexpected_failures=[r for r in real if 'unexpected_failure' in r],
        mask_only_refuted=all(r['mask_only_solution_refuted'] for r in negative),
        original_training_semantics_admitted=False, strict_causal_training_decision='NO-GO',
        formal_epochs=0, optimizer_updates=0, ETH_evaluated=False))


if __name__ == '__main__':
    main()
