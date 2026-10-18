"""Auditable selections and RAW-record counterfactual inputs, not ADE/FDE."""
from common import *
import copy
import numpy as np
import torch
from src.socialvae_causal.data import CausalDataset,array_sha
from src.risk_dynamic_graph import GraphConfig,build_graph
from src.risk_dynamic_graph.data import inference_inputs


def main():
    config = GraphConfig(**json.loads((CONF/'graph.json').read_text()))
    rows,counterfactuals = [],[]
    torch.set_num_threads(1)
    for fold in ('biwi_hotel','crowds_zara02'):
        manifest = ROOT/'configs/stage5p1/manifests'/fold/'B.json'
        train = CausalDataset(ROOT/'data/raw/eth_ucy',manifest,'train')
        # Uniformly spaced deterministic SOURCE TRAIN windows, not selected on
        # prediction errors or graph disagreements. No validation model outputs.
        for wi in np.unique(np.linspace(0,len(train.windows)-1,32,dtype=int)):
            window = train.windows[wi]
            inp = inference_inputs(train.records[window['recording']],window['start_frame'])
            graphs = {mode:build_graph(**inp,mode=mode,config=config) for mode in ('S','D','R')}
            d,r = graphs['D'],graphs['R']
            assert torch.equal(d['candidate_mask'],r['candidate_mask'])
            assert torch.equal(d['edge_mask'].sum(-1),r['edge_mask'].sum(-1))
            assert torch.equal(d['features'],r['features'])
            edges = {mode:int(g['edge_mask'].sum()) for mode,g in graphs.items()}
            rows.append(dict(fold=fold,sample_id=window['sample_id'],N=int(inp['ids'].shape[-1]),
                S_edges=edges['S'],D_edges=edges['D'],R_edges=edges['R'],
                D_R_changed_directed_edges=int((d['edge_mask'] & ~r['edge_mask']).sum()),
                candidate_mask_identical=True,per_receiver_budget_identical=True,message_features_identical=True,
                unknown_velocity_selected_R=int((r['edge_mask']&~r['velocity_known']).sum())))
        val = CausalDataset(ROOT/'data/raw/eth_ucy',manifest,'validation')
        wi,_ = val.indices[0]
        w = val.windows[wi]
        rec = val.records[w['recording']]
        a = inference_inputs(rec,w['start_frame'])
        changed = copy.deepcopy(rec)
        # Change all future coordinates/presence, including target GT and a
        # future-only identity; never change the eight historical records.
        for k in range(8,20):
            changed.by_frame[w['start_frame']+k*10] = {999999:np.array([100.,-100.],np.float32)}
        b = inference_inputs(changed,w['start_frame'])
        same = all(torch.equal(a[k],b[k]) for k in a)
        gs = []
        for mode in ('S','D','R'):
            ga = build_graph(**a,mode=mode,config=config)
            gb = build_graph(**b,mode=mode,config=config)
            gs.append(all(torch.equal(ga[k],gb[k]) for k in ('edge_mask','features','risk','candidate_mask')))
        counterfactuals.append(dict(fold=fold,sample_id=w['sample_id'],input_equal=same,all_graphs_equal=all(gs),
            original_history_SHA=array_sha(a['position'].numpy()),mutated_history_SHA=array_sha(b['position'].numpy()),
            mutated_future_support=[w['start_frame']+80,w['start_frame']+190],
            intervention='all future actor records removed; future-only ID999999 inserted',
            validation_predictions_computed=False,GT_errors_computed=False))
        assert same and all(gs)
    write_csv(OUT/'train_graph_selections.csv',rows)
    write_json(OUT/'raw_counterfactuals.json',counterfactuals)
    write_json(OUT/'selection_summary.json',dict(windows=len(rows),
        different_windows=sum(r['D_R_changed_directed_edges']>0 for r in rows),
        D_edges=sum(r['D_edges'] for r in rows),R_edges=sum(r['R_edges'] for r in rows),
        changed_edges=sum(r['D_R_changed_directed_edges'] for r in rows),
        source_only_no_future_errors=True,execution_commit=commit()))
    # Fully specified synthetic counterexample under frozen radius/sigma;
    # K=1 is a structural fixture, not the experimental K=8 hyperparameter.
    t=torch.arange(8,dtype=torch.float64)*.4
    p=torch.zeros(1,8,3,2,dtype=torch.float64)
    p[0,:,1,0]=1.+(t-t[-1])*.1
    p[0,:,2,0]=3.-(t-t[-1])
    inp=dict(position=p,valid=torch.ones(1,8,3,dtype=torch.bool),ids=torch.tensor([[10,20,30]]))
    from dataclasses import replace
    output={mode:build_graph(**inp,mode=mode,config=replace(config,max_neighbors=1)) for mode in ('D','R')}
    write_json(OUT/'synthetic_edge_witness.json',dict(
        input_position=p.tolist(),ids=inp['ids'].tolist(),example_config={**config.__dict__,'max_neighbors':1},
        D_selected_ids=inp['ids'][0,output['D']['edge_mask'][0,-1,0]].tolist(),
        R_selected_ids=inp['ids'][0,output['R']['edge_mask'][0,-1,0]].tolist(),
        last_frame_risk=output['R']['risk'][0,-1,0].tolist(),
        last_frame_tCPA=output['R']['t_cpa'][0,-1,0].tolist(),
        meaning='engineering distinction only, no evidence of prediction benefit'))


if __name__=='__main__':
    main()
