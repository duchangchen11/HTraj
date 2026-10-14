"""Inference-only graph controls for fixed best checkpoints, all three seeds.

Run a specified model only after its three runs exist. Separate result files
permit analysis while other GPU models train, without overwriting old evidence.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import sha256
from src.models.order_verification import build_order_model
from src.models.social_diagnostics_models import DIAGNOSTIC_MODELS
from src.evaluation.relation_controls import CONTROL_MODES,graph_control,control_counts
from src.training.order_verification import ROOT,PROTOCOL,datasets,cached_validation,evaluate
from src.training.social_diagnostics import csv_new


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['TOHI-H2','TOHI-H3','TOHI-NoOrder','Dynamic-GNN-RN'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    protocol=json.loads(PROTOCOL.read_text())
    _,val=datasets(protocol); batches=cached_validation(val,8)
    target=ROOT/'relations'/f'{args.model}.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError('Existing controls: '+str(target))
    paths={}
    for seed in protocol['seeds']:
        stage='stage2p5/round2' if args.model=='Dynamic-GNN-RN' else 'stage2p6/new'
        paths[seed]=Path('experiments')/stage/args.model/f'seed_{seed}/checkpoints/best.pt'
        assert paths[seed].exists(),paths[seed]
    rows=[]
    common=dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inference_runner_sha256=sha256(__file__),relation_engine_sha256=sha256('src/evaluation/relation_controls.py'),
        data_manifest_sha256=sha256(protocol['data_manifest']),subset_manifest_sha256=sha256(protocol['subset_manifest']),
        test_evaluated=False,retrained=False,negative_seed=protocol['negative_seed'],inference_device='cpu')
    for seed,path in paths.items():
        checkpoint=torch.load(path,weights_only=True,map_location='cpu')
        model=DIAGNOSTIC_MODELS[args.model](hidden_dim=64) if args.model=='Dynamic-GNN-RN' else build_order_model(args.model)
        model.load_state_dict(checkpoint['model']); model.eval()
        full=evaluate(model,batches,torch.device('cpu'),full_groups=False)
        for mode in CONTROL_MODES:
            counts={}
            with graph_control(model,mode,protocol['negative_seed']) as trace:
                result=evaluate(model,batches,torch.device('cpu'),reference=full['predictions'],full_groups=False)
            # Evaluate once per batch with the same deterministic graph control
            # to collect achieved edge/degree constraints independently of GT.
            with torch.no_grad(),graph_control(model,mode,protocol['negative_seed']) as trace:
                for batch in batches:
                    model(batch['observed'],batch['observed_valid'],batch['timestamps'])
                    actual=control_counts(trace['encoded'],trace['real'],trace['altered'])
                    for key,value in actual.items():
                        counts[key]=counts.get(key,0)+value
            denominator=max(counts['real_edges'],1)
            row=dict(model=args.model,seed=seed,checkpoint='best',best_epoch=checkpoint['epoch'],case=mode,
                status='executed',ADE_m=result['ADE_m'],FDE_m=result['FDE_m'],
                delta_ADE_vs_full=result['ADE_m']-full['ADE_m'],delta_FDE_vs_full=result['FDE_m']-full['FDE_m'],
                **result['diagnostics'],**counts,removed_real_edge_fraction=counts['removed_edges']/denominator,
                added_vs_real_edge_fraction=counts['added_edges']/denominator,
                theoretical_local_max_changed_fraction=counts['possible_local_changed_edges']/denominator,
                outside_radius_fraction=counts['control_outside_radius_edges']/max(counts['control_edges'],1),
                radius_constraint='relaxed' if mode in ['scene_random','scene_max_change','degree_swap_global'] else 'preserved',
                incoming_degree_constraint='preserved' if mode in ['full','radius_max_change','scene_random','scene_max_change','degree_swap_global'] else 'reduced',
                outgoing_degree_constraint='preserved' if mode in ['full','degree_swap_global'] else 'not_preserved',
                model_training_commit=checkpoint['config']['source_commit'],checkpoint_sha256=sha256(path),**common,
                interpretation='fixed checkpoint OOD intervention; not independent training or test performance')
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['model','seed','case','ADE_m','delta_ADE_vs_full','removed_real_edge_fraction','outside_radius_fraction']}),flush=True)
    csv_new(target,rows)


if __name__=='__main__':
    main()
