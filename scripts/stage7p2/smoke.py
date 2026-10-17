"""Official one-window train() execution, with real loader and original losses.

The window list is restricted for engineering smoke, never a complete epoch.
No network, attention, decoder, loss or sampler is replaced.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ast
import gc
import io
import json
import os
import subprocess
import time
import traceback
import numpy as np
import torch
import yaml
from src.agentformer_benchmark.official import install_compatibility, configuration, SOURCE, CACHE, DATA
from src.socialvae_baseline.artifacts import json_new, state_info
from src.data.preprocessing import sha256
from scripts.stage7p1.runtime import ResourceMeter

OUT = Path('results/tables/stage7p2')


def one_original_train_call(model, cfg, case):
    from data.dataloader import data_generator
    from utils.utils import AverageMeter
    log = io.StringIO()
    generator = data_generator(cfg, log, split='train', phase='training')
    si = generator.sequence_to_load.index(case['sequence'])
    pre = generator.sequence[si]
    index = sum(generator.num_sample_list[:si])+case['last_observed_frame']-7-pre.init_frame
    packet = pre(case['last_observed_frame'])
    assert len(packet['valid_id']) == case['people']
    original_slots = generator.num_total_samples
    generator.sample_list = [int(index)]
    generator.num_total_samples = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.decay_step, gamma=cfg.decay_gamma)
    observations = {}
    compute = model.compute_loss
    def checked_loss():
        loss, terms, unweighted = compute()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite original loss; no optimizer update allowed')
        observations.update(loss=float(loss.detach()), terms=terms, unweighted=unweighted)
        return loss, terms, unweighted
    model.compute_loss = checked_loss
    def check_gradient(opt, args, kwargs):
        gradients = [p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        observations['gradient_L2'] = sum(float(g.double().square().sum()) for g in gradients)**.5
        observations['gradient_parameters'] = sum(p.numel() for p in model.parameters() if p.grad is not None)
        observations['gradients_finite'] = True
    hook = optimizer.register_step_pre_hook(check_gradient)
    tree = ast.parse((SOURCE/'train.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    namespace = dict(time=time, generator=generator, model=model, optimizer=optimizer,
        scheduler=scheduler, cfg=cfg, AverageMeter=AverageMeter, tb_ind=0)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE/'train.py'), 'exec'), namespace)
    model.train()
    namespace['train'](0)
    hook.remove()
    model.compute_loss = compute
    observations.update(optimizer_updates=1, complete_epochs=0,
        original_candidate_frame_slots=int(original_slots), processed_frame_slots=1,
        model_context_people=int(model.data['agent_num']), scheduler_state=scheduler.state_dict(),
        original_train_function_SHA256=sha256(SOURCE/'train.py'), loader_log=log.getvalue())
    assert observations['model_context_people'] == case['people']
    return packet, optimizer, scheduler, observations


def infer(model, packet):
    model.eval()
    model.set_data(packet)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    with torch.no_grad():
        p, _ = model.inference(mode='infer', sample_num=20)
    assert p.shape == (len(packet['valid_id']), 20, 12, 2)
    assert torch.isfinite(p).all()
    return p.detach().cpu()


def main():
    install_compatibility()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    from model.model_lib import model_dict
    from easydict import EasyDict
    audit = json.load(open(OUT/'official_data_audit.json'))
    assert json.load(open(OUT/'evaluation_equivalence.json'))['official_eval_CLI_passed']
    workspace = (CACHE/'smoke_workspace').resolve()
    (workspace/'cfg').mkdir(parents=True, exist_ok=True)
    cases, updates = [], 0
    execution_commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    for number, case in enumerate(audit['smoke_cases']):
        row = dict(case, complete_epochs=0, optimizer_updates=0, pretrained_initialization=False,
            execution_commit=execution_commit, stages={}, context_truncated=False)
        print('Official real smoke', case['fold'], case['sequence'], case['last_observed_frame'], 'N', case['people'], flush=True)
        base_cfg = configuration(case['fold'])
        identifier = f'stage7p2_{case["fold"]}_case{number}_pre'
        serializable_cfg = json.loads(json.dumps(base_cfg))
        serializable_cfg['results_root_dir'] = str(workspace/'results')
        with (workspace/'cfg'/(identifier+'.yml')).open('x') as f:
            yaml.safe_dump(serializable_cfg, f)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        base = model_dict['agentformer'](base_cfg)
        base.set_device(torch.device('cuda'))
        before = state_info(base)
        try:
            with ResourceMeter() as meter:
                packet, optimizer, scheduler, observation = one_original_train_call(base, base_cfg, case)
                after = state_info(base)
                assert before['parameter_state_sha256'] != after['parameter_state_sha256']
                path = workspace/'results'/identifier/'models/model_0001.p'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    torch.save(dict(model_dict=base.state_dict(), opt_dict=optimizer.state_dict(),
                        scheduler_dict=scheduler.state_dict(), epoch=0, engineering_updates=1), f)
                base.zero_grad(set_to_none=True)
                expected = infer(base, packet)
                restored = model_dict['agentformer'](base_cfg)
                restored.set_device(torch.device('cuda'))
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                restored.load_state_dict(checkpoint['model_dict'], strict=True)
                restored_optimizer = torch.optim.Adam(restored.parameters(), lr=base_cfg.lr)
                restored_optimizer.load_state_dict(checkpoint['opt_dict'])
                restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=base_cfg.decay_step, gamma=base_cfg.decay_gamma)
                restored_scheduler.load_state_dict(checkpoint['scheduler_dict'])
                recovered = infer(restored, packet)
                torch.testing.assert_close(expected, recovered, atol=0, rtol=0)
                observation.update(checkpoint=str(path), checkpoint_sha256=sha256(path),
                    strict_model_optimizer_scheduler_restore=True, prediction_restore_max_difference=float((expected-recovered).abs().max()),
                    before_state=before, after_state=after, restored_optimizer_state_entries=len(restored_optimizer.state))
                del restored, checkpoint, restored_optimizer, restored_scheduler
            observation['resource'] = meter.result
            row['stages']['CVAE'] = observation
            updates += 1
            row['optimizer_updates'] += 1
            del optimizer, scheduler
            base.data = None
            base.set_device(torch.device('cpu'))
            del base
            gc.collect()
            torch.cuda.empty_cache()
            # Only cfg/checkpoint reference paths differ; this is an explicitly
            # named ONE-UPDATE engineering base, not an author trained epoch.
            cfg_dlow = configuration(case['fold'], stage='dlow')
            cfg_dlow.pred_cfg, cfg_dlow.pred_epoch = identifier, 1
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                dlow = model_dict['dlow'](cfg_dlow)
            finally:
                os.chdir(previous)
            dlow.set_device(torch.device('cuda'))
            frozen_before = state_info(dlow.pred_model[0])
            with ResourceMeter() as meter:
                packet, opt, schedule, observation = one_original_train_call(dlow, cfg_dlow, case)
                assert state_info(dlow.pred_model[0])['parameter_state_sha256'] == frozen_before['parameter_state_sha256']
                location = workspace/f'dlow_case_{number}.pth'
                with location.open('xb') as f:
                    torch.save(dict(model_dict=dlow.state_dict(), opt_dict=opt.state_dict(), scheduler_dict=schedule.state_dict(),
                        CVAE_checkpoint=str(path), CVAE_checkpoint_sha256=sha256(path), epoch=0, engineering_updates=1), f)
                dlow.zero_grad(set_to_none=True)
                dlow.pred_model[0].zero_grad(set_to_none=True)
                expected = infer(dlow, packet)
                os.chdir(workspace)
                try:
                    restored = model_dict['dlow'](cfg_dlow)
                finally:
                    os.chdir(previous)
                restored.set_device(torch.device('cuda'))
                checkpoint = torch.load(location, map_location='cpu', weights_only=False)
                restored.load_state_dict(checkpoint['model_dict'], strict=True)
                restored_optimizer = torch.optim.Adam(restored.parameters(), lr=cfg_dlow.lr)
                restored_optimizer.load_state_dict(checkpoint['opt_dict'])
                restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=cfg_dlow.decay_step, gamma=cfg_dlow.decay_gamma)
                restored_scheduler.load_state_dict(checkpoint['scheduler_dict'])
                recovered = infer(restored, packet)
                torch.testing.assert_close(expected, recovered, atol=0, rtol=0)
                observation.update(checkpoint=str(location), checkpoint_sha256=sha256(location),
                    strict_model_optimizer_scheduler_restore=True, base_CVAE_parameters_unchanged=True,
                    base_CVAE_checkpoint=str(path), base_CVAE_checkpoint_sha256=sha256(path),
                    prediction_restore_max_difference=float((expected-recovered).abs().max()),
                    registered_parameters=sum(p.numel() for p in dlow.parameters()),
                    separate_CVAE_parameters=frozen_before['parameter_count'])
                del restored, restored_optimizer, restored_scheduler, checkpoint
            observation['resource'] = meter.result
            row['stages']['DLow'] = observation
            row['optimizer_updates'] += 1
            updates += 1
            row['passed'] = True
            del dlow, opt, schedule
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            row['passed'] = False
            row['failure'] = traceback.format_exc()
            if hasattr(meter, 'result'):
                row['failure_resource'] = meter.result
            json_new(OUT/f'smoke/case_{number}.json', row)
            cases.append(row)
            break  # No retuning, context truncation, or silent retry.
        json_new(OUT/f'smoke/case_{number}.json', row)
        cases.append(row)
    json_new(OUT/'smoke_summary.json', dict(cases=cases, all_passed=len(cases) == 6 and all(r['passed'] for r in cases),
        actual_optimizer_updates=updates, maximum_optimizer_updates=12, complete_epochs=0,
        ETH_test_scores_computed=False, model_core_modified=False,
        official_full_train_function_executed=True, generator_budget_restricted_to_one_fixed_frame=True,
        source_SHA256=sha256(__file__), execution_commit=execution_commit))
    print('Engineering smoke complete:', len(cases), 'cases;', updates, 'updates; no full epoch', flush=True)


if __name__ == '__main__':
    main()
