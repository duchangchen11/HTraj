"""Observed-target cohort diagnostics; no GT coordinates or masks used."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
from collections import defaultdict
import numpy as np
import torch
from src.training.order_verification import datasets
from src.training.social_diagnostics import csv_new
from src.data.preprocessing import sha256,write_json_new
from src.evaluation.relation_routing import cache_validation
from src.models.relation_routing import build_routing_model


@torch.no_grad()
def main():
    torch.set_num_threads(1);p=json.loads(Path('configs/stage3/protocol.json').read_text());_,val=datasets(p);batches=cache_validation(val);rows=[]
    for name in p['new_models']:
        for endpoint in ('best','last'):
            path=Path('experiments/stage3/new')/name/f'seed_17/checkpoints/{endpoint}.pt';model=build_routing_model(name)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);model.eval();values=defaultdict(lambda:defaultdict(list))
            for batch in batches:
                _,d=model(batch['observed'],batch['observed_valid'],batch['timestamps'],return_debug=True)
                b,T,n,H=d['hidden'].shape;messages=torch.zeros(b*n);entropy=torch.zeros(b*n);last_degree=torch.zeros(b*n)
                for t,f in enumerate(d['graph_frames']):
                    messages.index_add_(0,f.target,torch.ones_like(f.weight))
                    if t==7:
                        entropy.index_add_(0,f.target,-f.weight*f.weight.clamp_min(1e-12).log());last_degree.index_add_(0,f.target,torch.ones_like(f.weight))
                quantities=dict(state_L2=d['order_states'][:,-1,:,0].norm(dim=-1),
                    message_L2=d['propagation_parts']['message'][:,-1,:,0].norm(dim=-1),carry_L2=d['propagation_parts']['carry'][:,-1,:,0].norm(dim=-1),
                    gate=d['residual_gate'],correction_L2=(d['fused_context']-d['hidden'][:,-1]).norm(dim=-1),
                    support=d['order_support'][:,-1,:,0].float(),messages_history=messages.reshape(b,n),neighbors_cutoff=last_degree.reshape(b,n),
                    routing_entropy=entropy.reshape(b,n),normalized_entropy=torch.where(last_degree>1,entropy/last_degree.clamp_min(2).log(),torch.zeros_like(entropy)).reshape(b,n))
                for group,mask in batch['groups'].items():
                    for k,v in quantities.items():values[group][k].extend(v[mask].tolist())
            for group,items in values.items():
                row=dict(model=name,seed=17,endpoint=endpoint,group=group,observed_targets=len(items['gate']),checkpoint_sha256=sha256(path))
                for key,v in items.items():
                    if v:row.update({key+'_mean':float(np.mean(v)),key+'_p05':float(np.quantile(v,.05)),key+'_p50':float(np.quantile(v,.5)),key+'_p95':float(np.quantile(v,.95))})
                rows.append(row)
    csv_new('results/tables/stage3/cohort_state_diagnostics.csv',rows)
    write_json_new('results/tables/stage3/cohort_state_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        history_only_masks=True,future_gt_used=False,test_evaluated=False,backend='CPU auxiliary latent diagnostics; main selected metrics CUDA',rows=len(rows)))


if __name__=='__main__':main()
