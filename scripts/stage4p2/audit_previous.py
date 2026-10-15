"""Inference-only factorial of old Full weights and BN buffers, without writes to old assets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess,time
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.public_training.model import make_model
from src.public_training.engine import evaluate


def main():
    torch.set_num_threads(1);root=Path('results/tables/stage4p2');start=time.perf_counter()
    checkpoints={e:Path(f'experiments/stage4p1/Full/epoch{e:03}.pt') for e in (29,30)}
    hashes={str(p):sha256(p) for p in checkpoints.values()}
    cache=Path('data/processed/stage4p1/A_windows.pt');assert sha256(cache)=='4f5a72f7ca08f9021d199f401da9f1136ccb89150c1814990e4bbf998a0104a1'
    samples=torch.load(cache,weights_only=False)['validation'];assert len(samples)==921
    states={e:torch.load(p,weights_only=False)['model'] for e,p in checkpoints.items()};rows=[]
    for weights in (29,30):
        for buffers in (29,30):
            model=make_model('Full').cuda()
            mixed={k:(states[buffers][k] if k.endswith(('running_mean','running_var','num_batches_tracked')) else v) for k,v in states[weights].items()}
            model.load_state_dict(mixed,strict=True);metrics,_=evaluate(model,samples)
            rows.append(dict(weight_epoch=weights,BN_buffer_epoch=buffers,**metrics));print(rows[-1],flush=True)
    # Reproduced original endpoints must match the already frozen historical curves.
    import csv
    curves=list(csv.DictReader(Path('results/tables/stage4p1/Full/curve_30.csv').open()))
    for e in (29,30):
        old=curves[e-1];current=next(r for r in rows if r['weight_epoch']==e and r['BN_buffer_epoch']==e)
        for metric in ('NLL','K1_distribution_mean_ADE','K1_distribution_mean_FDE'):
            # Original curve names have a validation prefix in some versions.
            key=metric if metric in old else 'validation_'+metric
            assert abs(float(old[key])-current[metric])<1e-6,(e,metric,old.keys())
    csv_new(root/'previous_full_weight_BN_factorial.csv',rows)
    model=make_model('Full');inactive=[(n,p.numel()) for n,p in model.named_parameters() if n.startswith(('tpcnns.4.','prelus.4.'))]
    for path,digest in hashes.items():assert sha256(path)==digest
    write_json_new(root/'previous_audit.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seconds=time.perf_counter()-start,cache_sha256=sha256(cache),checkpoint_hashes=hashes,checkpoints_unchanged=True,
        parameters=sum(p.numel() for p in model.parameters()),dormant_parameters=inactive,
        active_allocated_parameters=sum(p.numel() for p in model.parameters())-sum(n for _,n in inactive),
        scope='all old921 Zara2 windows; inference-only weight x BN buffer swap, not a retraining ablation',ETH_evaluated=False))


if __name__=='__main__':main()
