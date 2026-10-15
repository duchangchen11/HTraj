"""Official version history, strict EqMotion compatibility and bounded assets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import collections
import importlib
import platform
import subprocess
import torch
from scripts.stage5.common import ROOT, OUT, CACHE, COMMITS, git, sha, new_json, new_csv, export_blob, check_non_eth_asset


def main():
    records = []
    paths = ['eth_ucy/model_t.py', 'eth_ucy/gcl_t.py', 'main_eth_diverse.py']
    for commit in git('eqmotion', 'rev-list', '--all', '--reverse').decode().splitlines():
        tree = git('eqmotion', 'ls-tree', commit, *paths).decode().splitlines()
        blobs = {row.split('\t')[1]: row.split()[2] for row in tree}
        records.append(dict(commit=commit, **{p: blobs.get(p, '') for p in paths}))
    new_csv(OUT/'eqmotion_history.csv', records)
    unique = {tuple(r[p] for p in paths) for r in records if r[paths[0]]}
    assert len(unique) == 1, 'New official code found: audit manually before proceeding'
    # Use only documented fold dimensions, not checkpoint-driven architecture edits.
    sys.path.insert(0, str(CACHE/'eqmotion'/COMMITS['eqmotion']))
    EqMotion = importlib.import_module('eth_ucy.model_t').EqMotion
    loads = []
    for fold, channel in [('hotel', 64), ('zara1', 128)]:
        ckpt = ROOT/'data/external/stage4/assets/eqmotion/eth_ucy/saved_models'/(fold+'_ckpt.pth.tar')
        weights = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
        torch.manual_seed(17)
        model = EqMotion(8, 2, 64, 8, channel, 12, device='cpu', n_layers=4, recurrent=True)
        expected = model.state_dict()
        missing = sorted(set(expected)-set(weights))
        extra = sorted(set(weights)-set(expected))
        shape = {k:dict(model=list(expected[k].shape),checkpoint=list(weights[k].shape))
                 for k in expected.keys() & weights.keys() if expected[k].shape != weights[k].shape}
        strict_passed = False
        error = None
        try:
            model.load_state_dict(weights, strict=True)
            strict_passed = True
        except RuntimeError as e:
            error = str(e)
        assert not strict_passed, 'Official compatibility changed; review before any forward'
        loads.append(dict(fold=fold, official_channels=channel, official_category_num=model.category_num,
            checkpoint_sha256=sha(ckpt), checkpoint_path=str(ckpt.relative_to(ROOT)),
            model_parameters=sum(p.numel() for p in model.parameters()), expected_keys=len(expected), saved_keys=len(weights),
            missing_keys=missing, unexpected_keys=extra, shape_mismatches=shape,
            unexpected_prefix_counts=dict(collections.Counter(k.split('.')[0] for k in extra)),
            strict_passed=strict_passed, strict_error=error, forward_executed=False))
        print(f'EqMotion {fold}: strict_load_passed=False (strict=True); missing={len(missing)}, unexpected={len(extra)}, shapes={len(shape)}', flush=True)
    new_json(OUT/'eqmotion_strict_loading.json', dict(history_commits=len(records),unique_ETH_UCY_implementations=len(unique),
        refs=git('eqmotion','show-ref').decode(),code_origin_commit=next(r['commit'] for r in records if r[paths[0]]),
        configs=loads,checkpoint_forward=False,ETH_evaluated=False,stop='No reachable official matching implementation; no weight repair or non-strict load'))

    assets = []
    for name in ['models/hotel/ckpt-best','models/zara02/ckpt-best','data/hotel/test/biwi_hotel.txt',
                 'data/zara02/test/crowds_zara02.txt','data/zara02/train/crowds_zara01.txt']:
        check_non_eth_asset(name)
        assets.append(export_blob('socialvae', COMMITS['socialvae'], name, ROOT/'experiments/stage5/assets/socialvae'/name))
    new_json(OUT/'socialvae_assets.json', assets)
    new_json(OUT/'sgcn_history_audit.json',dict(current_head=git('sgcn','rev-parse','HEAD').decode().strip(),
        archived_code_commit=COMMITS['sgcn'],current_model_exists=bool(git('sgcn','ls-tree','HEAD','model.py').strip()),
        historical_files=git('sgcn','ls-tree','-rl',COMMITS['sgcn'],'model.py','train.py','test.py','checkpoint','checkpoints').decode(),
        source_history=git('sgcn','log','--all','--format=%H %ad %s','--date=short','--','model.py','train.py','test.py').decode(),
        status='Static analysis only; no model instantiation, checkpoint load or forward'))
    new_json(OUT/'audit_environment.json', dict(python=sys.version, platform=platform.platform(),torch=torch.__version__,
        cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),base_commit='e6e9700eba015140d00e989a2aaaf7d7a88bd00d',
        audit_script_sha256=sha(__file__),sources=COMMITS,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


if __name__ == '__main__':
    main()
