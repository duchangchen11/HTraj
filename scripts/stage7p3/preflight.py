"""Read-only inputs/loading preflight. No test predictions, no test errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contextlib
import io
import json
import os
import platform
import subprocess
import torch
import numpy as np
from src.agentformer_official_evaluation.runtime import *
from src.socialvae_baseline.artifacts import json_new


def load_pair(case, gpu=False):
    from model.model_lib import model_dict
    from utils.config import Config
    from utils.utils import prepare_seed
    cfg = Config(case['DLow_config'])
    assert cfg.pred_cfg == case['CVAE_config'] and cfg.pred_epoch == case['CVAE_checkpoint_index']
    assert cfg.sample_k == 20 and cfg.traj_scale == (2 if case['fold'] == 'eth' else 1)
    prepare_seed(cfg.seed)
    model = model_dict['dlow'](cfg)
    cp = torch.load(case['DLow_checkpoint'], map_location='cpu', weights_only=False)
    check = model.load_state_dict(cp['model_dict'], strict=True)
    assert not check.missing_keys and not check.unexpected_keys
    pre = torch.load(case['CVAE_checkpoint'], map_location='cpu', weights_only=False)['model_dict']
    actual = model.pred_model[0].state_dict()
    assert pre.keys() == actual.keys()
    assert all(torch.equal(pre[k], actual[k]) for k in pre)
    assert all(torch.isfinite(v).all() for v in [*model.parameters(), *model.pred_model[0].parameters()])
    model.set_device(torch.device('cuda:0' if gpu else 'cpu'))
    model.eval()
    return cfg, model


def main():
    inputs = verify_inputs()
    install_compatibility()
    torch.set_num_threads(1)
    workspace()
    original = Path.cwd()
    cases = []
    os.chdir(WORK)
    try:
        from data.dataloader import data_generator
        for case in protocol()['pairs']:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                cfg, model = load_pair(case)
                generator = data_generator(cfg, capture, split='test', phase='testing')
            cases.append(dict(fold=case['fold'], strict_CVAE=True, strict_DLow=True,
                exact_CVAE_tensor_equality=True, seed=cfg.seed, traj_scale=cfg.traj_scale,
                sequences=generator.sequence_to_load, candidate_frame_slots=int(generator.num_total_samples),
                model_parameters_including_unregistered_CVAE=sum(p.numel() for p in model.parameters())+
                    sum(p.numel() for p in model.pred_model[0].parameters()),
                loader_log=capture.getvalue(), test_predictions=0, test_errors_read=False))
            del model
    finally:
        os.chdir(original)
    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    json_new(OUT/'preflight.json', dict(inputs=inputs, cases=cases, no_test_errors=True, no_optimizer=True,
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, GPU=smi,
        cudnn_version=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        official_source_unchanged=True, compatibility='Runtime _LinearWithBias alias and np.int alias only; no source/core changes'))
    print('Preflight passed, five strict pairs and official test loaders; zero predictions/errors', flush=True)


if __name__ == '__main__':
    main()
