"""Bounded evaluation of frozen SocialVAE; wrappers, not a new architecture.

Official offline loader results are diagnostic. Raw-input causality is checked
separately: truncating an already preprocessed tensor cannot establish it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import time
import numpy as np
import torch
from scripts.stage5.common import ROOT, OUT, CACHE, COMMITS, sha, new_json, new_csv, metrics

SOURCE = CACHE/'socialvae'/COMMITS['socialvae']
ASSETS = ROOT/'experiments/stage5/assets/socialvae'
PLAN = ROOT/'configs/stage5/functional_plan.json'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE/filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def loader_for(cls, path, device, metadata=False):
    # Call the unchanged official load/collate functions serially. Avoid starting
    # one process per file; selection, feature construction and padding are unchanged.
    d = object.__new__(cls)
    d.ob_horizon, d.pred_horizon, d.horizon, d.frameskip = 8, 12, 20, 1
    d.device, d.batch_first = device, False
    d.flip = d.rotate = d.scale = False
    d.rng = np.random.RandomState(17)
    keys = []
    if metadata:
        lines, start = __import__('inspect').getsourcelines(cls.load)
        append_line = next(start+i for i, line in enumerate(lines) if 'traj.append((hist,' in line)
        def trace(frame, event, arg):
            if frame.f_code == cls.load.__code__ and event == 'line' and frame.f_lineno == append_line:
                v = frame.f_locals
                keys.append(dict(sample_index=len(keys), target_id=int(v['idx'][v['i']]),
                    first_frame=int(v['time'][v['tid0']]), last_observed_frame=int(v['time'][v['tid0']+7]),
                    last_future_frame=int(v['time'][v['tid0']+19])))
            return trace
        sys.settrace(trace)
    try:
        d.data = cls.load(d, str(path), [])
    finally:
        if metadata:
            sys.settrace(None)
    assert d.data
    if metadata:
        assert len(keys) == len(d.data)
    return d, keys


def history_tensor_hash(item):
    return hashlib.sha256(np.ascontiguousarray(item).tobytes()).hexdigest()


def checks(model, x, y, neighbor):
    model.eval()
    with torch.no_grad():
        original = model(x, neighbor)
        repeated = model(x, neighbor)
        order = torch.arange(x.shape[1]-1, -1, -1, device=x.device)
        target_order = model(x[:, order], neighbor[:, order])[:, order]
        neighbor_order = model(x, neighbor.flip(2))
        changed_future = neighbor.clone()
        changed_future[8:] = float('nan')
        isolated = model(x, changed_future)
        none = model(x, None)
        empty = model(x, neighbor[:, :, :0])
        padded = model(x, torch.cat((neighbor, torch.full_like(neighbor[:, :, :1], 1e9)), 2))
        shift = x.new_tensor([3., -2.])
        xx, nn = x.clone(), neighbor.clone()
        xx[..., :2] += shift
        nn[..., :2] += shift
        translated = model(xx, nn)
        torch.manual_seed(1017)
        a = model(x, neighbor, 20)
        torch.manual_seed(1017)
        b = model(x, neighbor, 20)
        rows = [dict(check=name, max_abs_error=float((actual-expected).abs().max()),
                     tolerance=tol, passed=bool(torch.allclose(actual, expected, atol=tol, rtol=tol)))
                for name, actual, expected, tol in [
                    ('deterministic_repeat', repeated, original, 1e-6),
                    ('target_batch_permutation', target_order, original, 2e-5),
                    ('neighbor_index_permutation', neighbor_order, original, 2e-5),
                    ('preprocessed_future_neighbor_isolation', isolated, original, 0.),
                    ('empty_vs_none_neighbors', empty, none, 0.),
                    ('neighbor_padding', padded, original, 2e-5),
                    ('translation', translated, original+shift, 2e-5),
                    ('seeded_K20_repeat', a, b, 1e-6)]]
        # This is a fixed-checkpoint sensitivity diagnostic, not trained NoSocial.
        sensitivity = float((original-none).norm(dim=-1).mean())
    assert torch.isfinite(original).all() and torch.isfinite(a).all()
    assert all(r['passed'] for r in rows), rows
    return rows, sensitivity


def raw_loader_check(cls, model, device):
    """Same observed raw positions, different future of last-frame entrant."""
    samples = []
    for label, speed in [('a', .2), ('b', 1.)]:
        file = ROOT/'experiments/stage5/cache'/('causality_'+label+'.txt')
        raw = []
        for t in range(20):
            raw.append(f'{t*10} 0 {t*.1} 0\n')
            if t >= 7:
                raw.append(f'{t*10} 1 {1.7+(t-7)*speed} 0\n')
        with file.open('x') as f:
            f.writelines(raw)
        d, _ = loader_for(cls, file, device)
        samples.append(d.collate_fn([d.data[0]]))
    xa, ya, na = samples[0]
    xb, yb, nb = samples[1]
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert torch.equal(na[:8, ..., :2], nb[:8, ..., :2])
    diff = float((na[:8, ..., 2:4]-nb[:8, ..., 2:4]).abs().max())
    with torch.no_grad():
        pa, pb = model(xa, na), model(xb, nb)
    assert diff > 0
    return dict(observed_raw_positions_identical=True, target_future_identical=True,
        history_neighbor_velocity_max_difference=diff,
        forecast_mean_difference_m=float((pa-pb).norm(dim=-1).mean()),
        forecast_max_difference_m=float((pa-pb).norm(dim=-1).max()),
        expected_finding='Official extend() fills a first-appearance neighbor velocity from next frame; at observation boundary this can use future',
        audit_passed=True, deployment_causality_passed=False, synthetic=True)


def main():
    plan = json.loads(PLAN.read_text())['SocialVAE']
    assert not (OUT/'socialvae_execution.json').exists(), 'Never overwrite prior experiments'
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    net = module('stage5_official_socialvae', 'social_vae.py')
    data = module('stage5_official_socialvae_data', 'data.py')
    util = module('stage5_official_socialvae_utils', 'utils.py')
    scores, targets, structure, runs = [], [], [], []
    start_all = time.perf_counter()
    for fold in plan['folds']:
        checkpoint = ASSETS/'models'/fold/'ckpt-best'
        recording = {'hotel':'biwi_hotel', 'zara02':'crowds_zara02'}[fold]
        raw = ASSETS/'data'/fold/'test'/(recording+'.txt')
        model = net.SocialVAE(horizon=12, ob_radius=2, hidden_dim=256).to(device).eval()
        # Official best checkpoints contain model tensors and scalar metadata.
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        incompatible = model.load_state_dict(saved['model'], strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        before = sha(SOURCE/'social_vae.py')
        t0 = time.perf_counter()
        d, keys = loader_for(data.Dataloader, raw, device, metadata=True)
        data_seconds = time.perf_counter()-t0
        selected = d.data[:plan['samples_per_fold']]
        assert len(selected) == plan['samples_per_fold']
        x, y, neighbor = d.collate_fn(selected[:16])
        rows, social_sensitivity = checks(model, x, y, neighbor)
        structure.extend(dict(fold=fold, **row) for row in rows)
        if fold == 'hotel':
            new_json(OUT/'socialvae_raw_loader_causality.json', raw_loader_check(data.Dataloader, model, device))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        observed_new_neighbor_cases = 0
        for batch, j in enumerate(range(0, len(selected), plan['batch_size'])):
            x, y, neighbor = d.collate_fn(selected[j:j+plan['batch_size']])
            # This uses only observed positions to flag an uncertain derivative.
            seen = neighbor[:8, ..., :2].abs().amax(-1) < 1e8
            uncertain = seen[-1] & ~seen[-2]
            near = (neighbor[7, ..., :2]-x[7, :, None, :2]).norm(dim=-1) <= 2
            boundary = (uncertain & near).any(-1)
            observed_new_neighbor_cases += int(boundary.sum())
            with torch.no_grad():
                det = model(x, neighbor)
                torch.manual_seed(plan['sampling_seed_base']+batch)
                stochastic = model(x, neighbor, plan['stochastic_samples'])
                ade, fde = metrics(stochastic, y)
                da, df = metrics(det[None], y)
                official_ade, official_fde = util.ADE_FDE(stochastic, y)
                assert torch.equal(ade, official_ade) and torch.equal(fde, official_fde)
            for offset in range(x.shape[1]):
                idx = j+offset
                hist, future, nei = selected[idx]
                row = dict(fold=fold, recording=recording, **keys[idx],
                    history_sha256=history_tensor_hash(hist), future_sha256=history_tensor_hash(future),
                    neighbor_sha256=history_tensor_hash(nei), history_context_agents=int(seen[:, offset].any(0).sum()),
                    boundary_new_near_neighbor=bool(boundary[offset]),
                    K1_prior_mean_ADE_m=float(da[0, offset]), K1_prior_mean_FDE_m=float(df[0, offset]),
                    K1_sample_ADE_m=float(ade[0, offset]), K1_sample_FDE_m=float(fde[0, offset]),
                    K20_minADE_m=float(ade[:, offset].min()), K20_minFDE_m=float(fde[:, offset].min()),
                    minADE_sample=int(ade[:, offset].argmin()),minFDE_sample=int(fde[:, offset].argmin()),
                    sampling_seed=plan['sampling_seed_base']+batch)
                targets.append(row)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-t0
        fold_targets = [r for r in targets if r['fold'] == fold]
        for mode, a, f in [('K1_prior_mean', 'K1_prior_mean_ADE_m','K1_prior_mean_FDE_m'),
                           ('K1_first_sample','K1_sample_ADE_m','K1_sample_FDE_m'),
                           ('K20_noFPC','K20_minADE_m','K20_minFDE_m')]:
            scores.append(dict(fold=fold, mode=mode, target_windows=len(fold_targets),
                ADE_m=sum(r[a] for r in fold_targets)/len(fold_targets),
                FDE_m=sum(r[f] for r in fold_targets)/len(fold_targets),
                independent_minima=mode=='K20_noFPC', official_paper_reproduced=False))
        runs.append(dict(fold=fold, checkpoint_sha256=sha(checkpoint), data_sha256=sha(raw),
            official_dataset_target_windows=len(d.data), evaluated_target_windows=len(selected),
            strict_load=True, missing_keys=0, unexpected_keys=0,
            parameters=sum(p.numel() for p in model.parameters()), state_dict_keys=len(model.state_dict()),
            inference_seconds=seconds, samples_per_second=len(selected)/seconds, loader_seconds=data_seconds,
            peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            no_neighbor_intervention_mean_prediction_difference_m=social_sensitivity,
            boundary_first_appearance_near_neighbor_cases=observed_new_neighbor_cases,
            official_scores_read=False, FPC=False, source_unchanged=before==sha(SOURCE/'social_vae.py')))
        print(json.dumps(runs[-1]), flush=True)
        if fold == 'hotel':
            smoke_model = net.SocialVAE(12, 2, 256).to(device).train()
            torch.manual_seed(17)
            # Reset every parameter with construction under the prescribed seed.
            del smoke_model
            smoke_model = net.SocialVAE(12, 2, 256).to(device).train()
            train_raw = ASSETS/'data/zara02/train/crowds_zara01.txt'
            train_data, _ = loader_for(data.Dataloader, train_raw, device)
            bx, by, bn = train_data.collate_fn(train_data.data[:16])
            opt = torch.optim.Adam(smoke_model.parameters(), lr=.0005)
            original = {k:v.detach().clone() for k,v in smoke_model.state_dict().items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            loss = smoke_model.loss(*smoke_model(bx, by, bn))
            assert all(torch.isfinite(v) for v in loss.values())
            loss['loss'].backward()
            assert all(torch.isfinite(p.grad).all() for p in smoke_model.parameters() if p.grad is not None)
            grad = math.sqrt(sum(float(p.grad.square().sum()) for p in smoke_model.parameters() if p.grad is not None))
            opt.step()
            torch.cuda.synchronize()
            train_seconds = time.perf_counter()-t0
            update = math.sqrt(sum(float((v-original[k]).square().sum()) for k,v in smoke_model.state_dict().items()))
            assert update > 0 and grad > 0
            smoke_path = ROOT/'experiments/stage5/assets/socialvae/smoke_seed17_one_step.pth'
            assert not smoke_path.exists()
            torch.save(dict(model=smoke_model.state_dict(), optimizer=opt.state_dict(), steps=1), smoke_path)
            restored = net.SocialVAE(12, 2, 256).to(device)
            restored.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)['model'], strict=True)
            assert all(torch.equal(a, b) for a,b in zip(smoke_model.state_dict().values(),restored.state_dict().values()))
            new_json(OUT/'socialvae_training_smoke.json',dict(seed=17, steps=1, batch_targets=16, recording='crowds_zara01',
                initialization='from scratch', optimizer='Adam', learning_rate=.0005,
                losses={k:float(v.detach()) for k,v in loss.items()}, gradient_L2=grad, parameter_update_L2=update,
                step_seconds=train_seconds, peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                allocation_scope='includes the retained inference model and fresh training model',
                strict_restore=True, checkpoint_sha256=sha(smoke_path), training_data_sha256=sha(train_raw),
                measured_batch_size=16, official_training_batch_size=128, full_training_completed=False,
                convergence_verified=False, ETH_evaluated=False))
            del restored, smoke_model, opt, original, loss, train_data
        del model, d
        torch.cuda.empty_cache()
    new_csv(OUT/'socialvae_metrics.csv', scores)
    new_csv(OUT/'socialvae_target_errors.csv', targets)
    new_csv(OUT/'socialvae_structure_checks.csv', structure)
    new_json(OUT/'socialvae_execution.json', dict(runs=runs, plan_sha256=sha(PLAN),script_sha256=sha(__file__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        total_seconds=time.perf_counter()-start_all, functional_candidates=['EqMotion','SocialVAE'],
        source_files={f:sha(SOURCE/f) for f in ['social_vae.py','data.py','utils.py','main.py']},
        no_source_patches=True, ETH_evaluated=False, formal_training=False, paper_reproduced=False,
        protocol='official offline loader limited diagnostic, causality limitation explicitly audited'))


if __name__ == '__main__':
    main()
