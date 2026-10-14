"""Supplement path COUNT fractions with target SUPPORT, without changing old CSVs."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256
from src.models.ego_encoder import motion_inputs
from src.models.temporal_graph import select_candidates
from src.evaluation.path_identity import path_identity_counts
from src.training.order_verification import ROOT,PROTOCOL,datasets
from src.training.social_diagnostics import csv_new


def main():
    torch.set_num_threads(1); protocol=json.loads(PROTOCOL.read_text()); train,val=datasets(protocol)
    common=dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_runner_sha256=sha256(__file__),counter_sha256=sha256('src/evaluation/path_identity.py'),
        data_manifest_sha256=sha256(protocol['data_manifest']),subset_manifest_sha256=sha256(protocol['subset_manifest']),
        GT_used_for_statistics=False,test_evaluated=False)
    rows=[]; pooled=[]
    for split,data in [('train',train),('validation',val)]:
        totals={mode:{} for mode in ['strict','nondecreasing']}
        for index,batch in enumerate(DataLoader(data,batch_size=1,collate_fn=collate_trajectories)):
            encoded=motion_inputs(batch['observed'],batch['observed_valid'],batch['timestamps'])
            adj=np.stack([select_candidates(encoded['positions'][:,t],encoded['velocity'][:,t],
                encoded['continuity'][:,t],radius=4.,max_neighbors=8)[0][0].numpy() for t in range(8)])
            cutoff=batch['observed_valid'][0,-1].numpy()
            for mode in ['strict','nondecreasing']:
                a=path_identity_counts(adj,encoded['continuity'][0].numpy(),mode)
                masks=dict(q2_any=a['q2_total']>0,q2_three_distinct=a['q2_three_distinct']>0,
                    q2_return_only=(a['q2_total']>0)&(a['q2_three_distinct']==0),q3_any=a['q3_total']>0,
                    q3_four_distinct=a['q3_four_distinct']>0,q3_repeat_only=(a['q3_total']>0)&(a['q3_four_distinct']==0))
                counts={k:int((v&cutoff).sum()) for k,v in masks.items()}
                counts['cutoff_visible_target_windows']=int(cutoff.sum())
                for key,value in counts.items():
                    totals[mode][key]=totals[mode].get(key,0)+value
                rows.append(dict(split=split,mode=mode,subset_window_index=index,**counts,**common))
        for mode,counts in totals.items():
            fractions={k+'_visible_fraction':v/counts['cutoff_visible_target_windows'] for k,v in counts.items()
                       if k!='cutoff_visible_target_windows'}
            pooled.append(dict(split=split,mode=mode,**counts,**fractions,**common))
    csv_new(ROOT/'distinct_path_coverage_by_window.csv',rows)
    csv_new(ROOT/'distinct_path_coverage_summary.csv',pooled)
    print(json.dumps(pooled),flush=True)


if __name__=='__main__':
    main()
