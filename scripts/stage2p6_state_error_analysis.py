"""Descriptive state/error associations on every fixed best checkpoint.

No scales are changed. No norms/GT are used to select models or data. Quartiles
are exploratory summaries; correlations are not causal evidence of instability.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import sha256
from src.models.order_verification import MODEL_SPECS,build_order_model
from src.training.order_verification import ROOT,PROTOCOL,datasets,cached_validation
from src.training.social_diagnostics import csv_new


def ranks(values):
    values=np.asarray(values)
    order=np.argsort(values,kind='stable'); result=np.zeros(len(values),dtype=float)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:
            end+=1
        result[order[start:end]]=(start+end-1)/2.
        start=end
    return result


def correlations(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return None,None
    return float(np.corrcoef(x,y)[0,1]),float(np.corrcoef(ranks(x),ranks(y))[0,1])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True,choices=list(MODEL_SPECS))
    args=parser.parse_args(); torch.set_num_threads(1)
    protocol=json.loads(PROTOCOL.read_text())
    _,val=datasets(protocol); batches=cached_validation(val,8)
    directory=ROOT/'state_error'; directory.mkdir(parents=True,exist_ok=True)
    if (directory/f'{args.model}_correlations.csv').exists():
        raise FileExistsError('Existing state/error analysis')
    for seed in protocol['seeds']:
        assert (Path('experiments/stage2p6/new')/args.model/f'seed_{seed}/checkpoints/best.pt').exists()
    correlations_rows,quartile_rows,matrix_rows=[] ,[],[]
    common=dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inference_runner_sha256=sha256(__file__),data_manifest_sha256=sha256(protocol['data_manifest']),
        subset_manifest_sha256=sha256(protocol['subset_manifest']),test_evaluated=False,
        inference_device='cpu',scale_modified=False,interpretation='descriptive association, not scale-causality evidence')
    for seed in protocol['seeds']:
        path=Path('experiments/stage2p6/new')/args.model/f'seed_{seed}/checkpoints/best.pt'
        checkpoint=torch.load(path,weights_only=True,map_location='cpu')
        model=build_order_model(args.model); model.load_state_dict(checkpoint['model']); model.eval()
        fields={}; cohorts={name:[] for name in ['all','full8','partial','full8_low','full8_medium','full8_high']}
        def add(name,tensor,mask):
            fields.setdefault(name,[]).append(tensor[mask].double().numpy())
        with torch.no_grad():
            for batch in batches:
                pred,debug=model(batch['observed'],batch['observed_valid'],batch['timestamps'],return_debug=True)
                valid=batch['future_valid']; evaluable=valid.any(1) & batch['groups']['all']
                distance=(pred-batch['future']).norm(dim=-1)
                ade=(distance*valid).sum(1)/valid.sum(1).clamp_min(1)
                fde=distance[:,-1].clone(); fde[~valid[:,-1]]=float('nan')
                add('ADE_m',ade,evaluable); add('FDE_m',fde,evaluable)
                ego=debug['hidden'][:,-1].norm(dim=-1)
                add('ego_L2',ego,evaluable)
                add('gate',debug['residual_gate'],evaluable)
                add('correction_L2',(debug['fused_context']-debug['hidden'][:,-1]).norm(dim=-1),evaluable)
                for q in range(model.order):
                    norm=debug['order_states'][:,-1,:,q].norm(dim=-1)
                    add(f'q{q+1}_raw_L2',norm,evaluable)
                    add(f'q{q+1}_raw_to_ego',norm/ego.clamp_min(1e-8),evaluable)
                    add(f'q{q+1}_beta',debug['order_weights'][:,:,q],evaluable)
                    for part in ['before','message','carry']:
                        add(f'q{q+1}_{part}_L2',debug['propagation_parts'][part][:,-1,:,q].norm(dim=-1),evaluable)
                masks=dict(all=batch['groups']['all'],full8=batch['groups']['history_full8'],partial=batch['groups']['history_partial'],
                    full8_low=batch['groups']['full8__low_neighbors'],full8_medium=batch['groups']['full8__medium_neighbors'],
                    full8_high=batch['groups']['full8__high_neighbors'])
                for name,mask in masks.items():
                    cohorts[name].append(mask[evaluable].numpy())
        arrays={k:np.concatenate(v) for k,v in fields.items()}
        masks={k:np.concatenate(v) for k,v in cohorts.items()}
        provenance=dict(model=args.model,seed=seed,checkpoint='best',best_epoch=checkpoint['epoch'],
            model_training_commit=checkpoint['config']['source_commit'],checkpoint_sha256=sha256(path),**common)
        for name,mask in masks.items():
            for feature,x in arrays.items():
                if feature in ['ADE_m','FDE_m']:
                    continue
                for metric in ['ADE_m','FDE_m']:
                    use=mask & np.isfinite(x) & np.isfinite(arrays[metric])
                    pearson,spearman=correlations(x[use],arrays[metric][use])
                    correlations_rows.append(dict(**provenance,cohort=name,feature=feature,metric=metric,n=int(use.sum()),
                        feature_mean=float(x[use].mean()) if use.any() else None,feature_p95=float(np.quantile(x[use],.95)) if use.any() else None,
                        metric_mean=float(arrays[metric][use].mean()) if use.any() else None,pearson=pearson,spearman=spearman))
            for q in range(model.order):
                feature=f'q{q+1}_raw_L2'; x=arrays[feature]
                if not mask.any():
                    continue
                cuts=np.quantile(x[mask],[.25,.5,.75])
                bins=np.searchsorted(cuts,x,side='right')
                for index in range(4):
                    use=mask & (bins==index)
                    fde_mask=use & np.isfinite(arrays['FDE_m'])
                    quartile_rows.append(dict(**provenance,cohort=name,feature=feature,quartile=index+1,
                        q25=float(cuts[0]),q50=float(cuts[1]),q75=float(cuts[2]),n=int(use.sum()),
                        FDE_count=int(fde_mask.sum()),state_norm_mean=float(x[use].mean()) if use.any() else None,
                        ADE_m=float(arrays['ADE_m'][use].mean()) if use.any() else None,
                        FDE_m=float(arrays['FDE_m'][fde_mask].mean()) if fde_mask.any() else None,
                        bins_use_GT=False,bins_used_for_selection=False))
        torch.manual_seed(seed); initial=build_order_model(args.model)
        for q,(module,initial_module) in enumerate(zip(model.order_state.projections,initial.order_state.projections)):
            singular=np.linalg.svd(module.weight.detach().numpy(),compute_uv=False)
            initial_singular=np.linalg.svd(initial_module.weight.detach().numpy(),compute_uv=False)
            matrix_rows.append(dict(**provenance,order=q+1,spectral_norm=float(singular.max()),
                initial_spectral_norm=float(initial_singular.max()),frobenius_norm=float(np.linalg.norm(singular)),
                mean_singular_value=float(singular.mean())))
    csv_new(directory/f'{args.model}_correlations.csv',correlations_rows)
    csv_new(directory/f'{args.model}_quartiles.csv',quartile_rows)
    csv_new(directory/f'{args.model}_projection_spectra.csv',matrix_rows)
    print(args.model,'all seeds state/error analysis complete',flush=True)


if __name__=='__main__':
    main()
