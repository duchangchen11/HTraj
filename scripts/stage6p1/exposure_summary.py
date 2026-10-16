"""Summarize actual source-window and person exposure, not model selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from collections import defaultdict
import numpy as np
from src.socialvae_causal.io import read_csv,csv_new
from src.motion_sampling.policy import NAMES

ROOT=Path('results/tables/stage6p1')


def main():
    rows=[];people=[];actual=[]
    for rec in ['biwi_hotel','crowds_zara02']:
        pool=read_csv(ROOT/f'pools/{rec}_train.csv.gz')
        for condition in ['Uniform','Stratified']:
            source=ROOT/'runs'/rec/condition
            assert json.load(open(source/'selection.json'))['training_completed']
            curve=read_csv(source/'sampling_curve.csv')
            counts=read_csv(source/'epochs/epoch030_target_visits.csv.gz')
            for g,name in enumerate(NAMES):
                members=[int(r['target_index']) for r in pool if r['category']==name]
                c=np.array([int(counts[i]['cumulative_visits']) for i in members])
                sub=[r for r in curve if r['category']==name]
                actual.append(dict(recording=rec,condition=condition,category=name,pool_windows=len(members),
                    pool_person_IDs=len({(pool[i]['recording'],pool[i]['target_id']) for i in members}),
                    total_visits=int(c.sum()),actual_mean_visits_per_window=float(c.mean()) if len(c) else None,
                    min_visits_per_window=int(c.min()) if len(c) else None,max_visits_per_window=int(c.max()) if len(c) else None,
                    cumulative_unique_windows=int((c>0).sum()),cumulative_coverage=float((c>0).mean()) if len(c) else None,
                    mean_epoch_sample_fraction=float(np.mean([float(r['sampled_fraction']) for r in sub])),
                    mean_epoch_coverage=float(np.mean([float(r['coverage']) for r in sub])) if len(c) else None,
                    total_repeated_visits_within_epoch=sum(int(r['repeated_visits']) for r in sub),
                    mean_repeated_target_windows=float(np.mean([int(r['repeated_target_ID_windows']) for r in sub]))))
            by_id=defaultdict(list)
            for i,row in enumerate(pool):by_id[(row['recording'],row['target_id'])].append(i)
            for k,indices in by_id.items():
                visits=sum(int(counts[i]['cumulative_visits']) for i in indices)
                people.append(dict(fold=rec,condition=condition,source_recording=k[0],target_id=k[1],
                    source_target_windows=len(indices),actual_30epoch_visits=visits,
                    ever_visited_windows=sum(int(counts[i]['cumulative_visits'])>0 for i in indices)))
    csv_new(ROOT/'actual_class_sampling_summary.csv',actual)
    csv_new(ROOT/'actual_source_person_exposure.csv.gz',people)
    print('Actual class/window/recording-person exposure summarized without changing selection')


if __name__=='__main__':main()
