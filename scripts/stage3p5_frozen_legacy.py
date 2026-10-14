"""B legacy 1024/96 frozen three-seed re-evaluation, never a new fold result."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
import torch
from torch.utils.data import Subset,DataLoader
from src.data.dataset import TrajectoryDataset
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256,write_json_new
from src.evaluation.benchmark_audit import history_features
from src.training.benchmark_audit import evaluate,save_eval
from src.models.social_diagnostics_models import DIAGNOSTIC_MODELS
from src.models.relation_routing import HistoricalCV


def main():
    torch.set_num_threads(1);manifest='configs/data_manifests/stage2_v1/fold_ETH.json';subset='configs/stage2p5/subset_1024.json'
    ids=json.loads(Path(subset).read_text())['validation']['indices'];data=Subset(TrajectoryDataset('data/raw/eth_ucy',manifest,'validation'),ids);batches=[]
    for b in DataLoader(data,batch_size=8,collate_fn=collate_trajectories):b['groups'],b['history_features']=history_features(b);batches.append(b)
    root=Path('results/tables/stage3p5/B_legacy_1024_96_frozen_analysis');audit=[]
    for name in ('CV','PedBase','Dynamic-GNN-RN'):
        for seed in ([0] if name=='CV' else [17,42,2026]):
            out=root/name/f'seed_{seed}';out.mkdir(parents=True)
            if name=='CV':model=HistoricalCV();r=evaluate(model,batches,torch.device('cpu'),rows=True)
            else:
                origin=Path('results/tables/stage2p5/round2')/name/f'seed_{seed}';s=json.loads((origin/'summary.json').read_text());c=json.loads((origin/'config.json').read_text())
                assert c['data_manifest_sha256']==sha256(manifest) and c['subset_manifest_sha256']==sha256(subset)
                source=c['source_commit'];paths=subprocess.check_output(['git','ls-tree','-r','--name-only',source,'src'],text=True).splitlines()
                for p in paths:assert Path(p).read_bytes()==subprocess.check_output(['git','show',source+':'+p]),p
                checkpoint=Path('experiments/stage2p5/round2')/name/f'seed_{seed}/checkpoints/best.pt';assert sha256(checkpoint)==s['best_checkpoint_sha256']
                model=DIAGNOSTIC_MODELS[name](hidden_dim=64);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);r=evaluate(model,batches,torch.device('cpu'),rows=True)
                assert abs(r['ADE_m']-s['best_ADE_m'])<1e-6 and abs(r['FDE_m']-s['best_FDE_m'])<1e-6
                audit.append(dict(model=name,seed=seed,original_config=str(origin/'config.json'),original_summary=str(origin/'summary.json'),checkpoint_sha256=sha256(checkpoint),source_commit=source))
            save_eval(out,'best',r);write_json_new(out/'summary.json',dict(model=name,seed=seed,ADE_m=r['ADE_m'],FDE_m=r['FDE_m'],new_training=False,setting='B_legacy1024/96_frozen',ETH_evaluated=False))
    write_json_new(root/'compatibility_audit.json',dict(status='passed',runs=audit,data_manifest_sha256=sha256(manifest),subset_manifest_sha256=sha256(subset),new_training_runs=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),ETH_evaluated=False,metrics_tolerance_m=1e-6))


if __name__=='__main__':main()
