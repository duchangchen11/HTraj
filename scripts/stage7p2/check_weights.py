"""Strictly load BOTH author CVAE and DLow checkpoints, never score ETH."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contextlib
import io
import json
import os
import traceback
import torch
from src.agentformer_benchmark.official import install_compatibility, SOURCE, CACHE
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

OUT = Path('results/tables/stage7p2')


def main():
    install_compatibility()
    torch.set_num_threads(1)
    # Follow the author's train.py import order; importing dlow directly first
    # triggers its existing model_lib circular import.
    from model.model_lib import model_dict
    AgentFormer, DLow = model_dict['agentformer'], model_dict['dlow']
    from utils.config import Config
    acquisition = json.load(open(OUT/'official_checkpoint_acquisition.json'))
    if not acquisition['available']:
        json_new(OUT/'official_checkpoint_strict_loading.json', dict(available=False, all_folds_strict=False, cases=[]))
        return
    workspace = (CACHE/'strict_load_workspace').resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace/'cfg').symlink_to(SOURCE/'cfg', target_is_directory=True)
    models = (CACHE/'pretrained/extracted/results').resolve()
    for folder in models.iterdir():
        if 'nuscenes' in folder.name:
            continue
        destination = workspace/'results'/folder.name
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'models').symlink_to(folder/'models', target_is_directory=True)
    root = Path.cwd()
    cases = []
    os.chdir(workspace)
    try:
        for fold in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
            row = dict(fold=fold, strict=False, ETH_evaluated=False)
            try:
                cfg = Config(f'{fold}_agentformer')
                pre_cfg = Config(cfg.pred_cfg)
                pre_path = Path(pre_cfg.model_path % cfg.pred_epoch)
                full_path = sorted((models/f'{fold}_agentformer/models').glob('model_*.p'))[-1]
                cp_pre = torch.load(pre_path, map_location='cpu', weights_only=False)
                base = AgentFormer(pre_cfg)
                check_pre = base.load_state_dict(cp_pre['model_dict'], strict=True)
                assert not check_pre.missing_keys and not check_pre.unexpected_keys
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    full = DLow(cfg)  # Original constructor strictly restores its separate base.
                cp_full = torch.load(full_path, map_location='cpu', weights_only=False)
                check_full = full.load_state_dict(cp_full['model_dict'], strict=True)
                assert not check_full.missing_keys and not check_full.unexpected_keys
                assert all(torch.isfinite(p).all() for p in base.parameters())
                assert all(torch.isfinite(p).all() for p in full.parameters())
                assert set(base.state_dict()) == set(full.pred_model[0].state_dict())
                assert all(torch.equal(v, full.pred_model[0].state_dict()[k]) for k, v in base.state_dict().items())
                row.update(strict=True, CVAE_checkpoint=str(pre_path.resolve()), CVAE_sha256=sha256(pre_path),
                    CVAE_checkpoint_index=cfg.pred_epoch, CVAE_config=cfg.pred_cfg,
                    CVAE_config_sha256=sha256(SOURCE/f'cfg/eth_ucy/{fold}/{cfg.pred_cfg}.yml'),
                    CVAE_parameters=sum(p.numel() for p in base.parameters()),
                    CVAE_state_keys=len(base.state_dict()), DLow_checkpoint=str(full_path), DLow_sha256=sha256(full_path),
                    DLow_checkpoint_index=int(full_path.stem.split('_')[-1]), DLow_config=f'{fold}_agentformer',
                    DLow_config_sha256=sha256(SOURCE/f'cfg/eth_ucy/{fold}/{fold}_agentformer.yml'),
                    DLow_registered_parameters=sum(p.numel() for p in full.parameters()),
                    DLow_state_keys=len(full.state_dict()),
                    both_parts_required=True, wrapped_CVAE_is_not_in_DLow_state_dict=True,
                    full_parameters_including_external_CVAE=sum(p.numel() for p in full.parameters())+sum(p.numel() for p in base.parameters()),
                    constructor_log=capture.getvalue())
                del base, full, cp_pre, cp_full
            except Exception:
                row['failure'] = traceback.format_exc()
            cases.append(row)
    finally:
        os.chdir(root)
    json_new(OUT/'official_checkpoint_strict_loading.json', dict(available=True, cases=cases,
        official_source_commit='e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5',
        all_folds_strict=all(r['strict'] for r in cases), used_strict_False=False,
        ETH_test_scores_computed=False, no_benchmark_metrics_computed=True,
        archive_sha256=next(s['sha256'] for s in acquisition['download_steps'] if 'archive' in s),
        normalized_code_is_history_only='scene_orig derived only from pre_motion at the pinned source',
        source_SHA256=sha256(__file__)))
    print('Strict CVAE+DLow loading:', [(r['fold'], r['strict']) for r in cases], flush=True)


if __name__ == '__main__':
    main()
