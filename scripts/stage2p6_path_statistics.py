"""History-only identity statistics on exactly the fixed 1024/96 windows."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256,write_json_new
from src.models.ego_encoder import motion_inputs
from src.models.temporal_graph import select_candidates
from src.evaluation.path_identity import path_identity_counts
from src.training.order_verification import ROOT,PROTOCOL,datasets
from src.training.social_diagnostics import csv_new


def ratios(counts):
    def ratio(a,b):
        return float(counts[a]/counts[b]) if counts[b] else None
    return dict(q2_three_distinct_fraction=ratio('q2_three_distinct','q2_total'),
        q2_return_fraction=ratio('q2_return','q2_total'),
        q2_unique_triples_per_timed_walk=ratio('q2_unique_identity_triples','q2_total'),
        q2_nonreturn_pairs_multiple_middles_fraction=ratio('q2_nonreturn_pairs_multiple_middles','q2_nonreturn_endpoint_pairs'),
        q2_mean_distinct_middles_per_nonreturn_pair=ratio('q2_nonreturn_distinct_middle_sum','q2_nonreturn_endpoint_pairs'),
        q3_repeated_fraction=ratio('q3_repeated','q3_total'),q3_backtrack_fraction=ratio('q3_backtrack','q3_total'),
        q3_end_return_fraction=ratio('q3_end_return','q3_total'),q3_four_distinct_fraction=ratio('q3_four_distinct','q3_total'))


def main():
    torch.set_num_threads(1)
    protocol=json.loads(PROTOCOL.read_text())
    train,val=datasets(protocol)
    rows=[]; pooled=[]
    common=dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_runner_sha256=sha256(__file__),identity_counter_sha256=sha256('src/evaluation/path_identity.py'),
        data_manifest_sha256=sha256(protocol['data_manifest']),subset_manifest_sha256=sha256(protocol['subset_manifest']),
        test_evaluated=False,GT_used_for_statistics=False)
    for split,data in [('train',train),('validation',val)]:
        totals={mode:{} for mode in ['strict','nondecreasing']}
        targets=0
        for index,batch in enumerate(DataLoader(data,batch_size=1,collate_fn=collate_trajectories)):
            encoded=motion_inputs(batch['observed'],batch['observed_valid'],batch['timestamps'])
            adjacency=np.stack([select_candidates(encoded['positions'][:,t],encoded['velocity'][:,t],
                encoded['continuity'][:,t],radius=4.,max_neighbors=8)[0][0].numpy() for t in range(8)])
            cutoff=batch['observed_valid'][0,-1].numpy()
            targets+=int(cutoff.sum())
            for mode in ['strict','nondecreasing']:
                arrays=path_identity_counts(adjacency,encoded['continuity'][0].numpy(),mode)
                counts={k:int(v[cutoff].sum()) for k,v in arrays.items()}
                for key,value in counts.items():
                    totals[mode][key]=totals[mode].get(key,0)+value
                rows.append(dict(split=split,subset_window_index=index,sample_id=batch['sample_id'][0],
                    mode=mode,cutoff_visible_targets=int(cutoff.sum()),**counts,**ratios(counts),**common))
            if (index+1)%128==0:
                print(split,index+1,flush=True)
        for mode,counts in totals.items():
            pooled.append(dict(split=split,mode=mode,windows=len(data),cutoff_visible_target_windows=targets,
                **counts,**ratios(counts),**common))
    csv_new(ROOT/'path_identity_by_window.csv',rows)
    csv_new(ROOT/'path_identity_summary.csv',pooled)
    write_json_new(ROOT/'path_statistics_provenance.json',dict(**common,
        definition='timed edge walks surviving at cutoff; waiting does not multiply events; micro pooled over visible targets/window',
        not_independent_people=True,not_observed_social_causality=True,
        train_windows=1024,validation_windows=96))
    print(json.dumps(pooled,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
