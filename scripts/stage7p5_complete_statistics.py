"""Complete nullable-field aggregation and source/actor sensitivity, no new inference."""
import collections
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
from stage7p5_analyze import summarize


def read(path):
    with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in list(r.items()):
            if k in ('fold','sequence','speed_group','turn_group','density_group','GT_SHA256','history_SHA256'):continue
            if v=='':r[k]=None
            elif v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows


def main():
    out=ROOT/'results/tables/stage7p5';complete={};correlations=[];actor_rows=[];source_speed={};head_statistics=[];macro={}
    for fold in ('hotel','zara2'):
        rows=read(out/f'{fold}_targets.csv.gz')
        # The initial generic aggregator chose numeric columns from its first row.
        # Nullable direction columns could disappear; explicitly use the union here.
        keys=sorted({k for r in rows for k,v in r.items() if isinstance(v,(float,int))}-{ 'frame','person_id','context_people'})
        complete[fold]={};source_speed[fold]={}
        for field in ('all','sequence','speed_group','turn_group','density_group'):
            complete[fold][field]={}
            for label in (['all'] if field=='all' else sorted({r[field] for r in rows})):
                rr=rows if field=='all' else [r for r in rows if r[field]==label]
                complete[fold][field][label]=summarize(rr,keys)
        for seq in sorted({r['sequence'] for r in rows}):
            source_speed[fold][seq]={}
            for speed in ('stationary','low','normal','fast'):
                source_speed[fold][seq][speed]=summarize([r for r in rows if r['sequence']==seq and r['speed_group']==speed],keys)
        grouped=collections.defaultdict(list)
        for r in rows:grouped[r['sequence'],r['person_id']].append(r)
        actors=[]
        for (seq,person),rr in grouped.items():
            z=summarize(rr,keys)['window'];a=dict(fold=fold,sequence=seq,person_id=int(person),windows=len(rr),**z)
            actors.append(a);actor_rows.append(a)
        for seq in ['all']+sorted({r['sequence'] for r in rows}):
            aa=actors if seq=='all' else [a for a in actors if a['sequence']==seq]
            for x in ('history_speed_mps','history_speed_sd_mps','history_fit_RMS_m','history_neighbors_2m','nearest_history_neighbor_m','local_relative_speed_mps','local_closing_mps'):
                for y in ('ADE_oracle_ADE','ADE_oracle_ADE_over_GT_path_floor0.5'):
                    pairs=[(a[x],a[y]) for a in aa if a.get(x) is not None and a.get(y) is not None]
                    value=None
                    if len(pairs)>=3:
                        u,v=np.array(pairs).T
                        if np.std(u)>0 and np.std(v)>0:value=float(spearmanr(u,v).statistic)
                    correlations.append(dict(fold=fold,sequence=seq,predictor=x,outcome=y,persons=len(pairs),spearman_r=value,inference='descriptive only; no independent-window p-value'))
        groups=complete[fold]['sequence'].values()
        macro[fold]={k:float(np.mean([z['window'][k] for z in groups if k in z['window']])) for k in ('ADE_oracle_ADE','FDE_oracle_FDE','ADE_oracle_speed_bias_mps','all_heads_speed_bias_mps')}
        hh=[h for h in json.load(open(out/'head_contributions.json')) if h['fold']==fold]
        for field,label in sorted({(h['group_field'],h['group']) for h in hh}):
            rr=sorted([h for h in hh if h['group_field']==field and h['group']==label],key=lambda r:r['head'])
            for weight in ('window','person_equal'):
                for metric in ('ADE_credit','FDE_credit'):
                    a=np.array([r[weight][metric] for r in rr]);order=np.argsort(-a)
                    head_statistics.append(dict(fold=fold,group_field=field,group=label,weight=weight,metric=metric,
                        top_head=int(order[0]),top_fraction=float(a[order[0]]),top4_fraction=float(a[order[:4]].sum()),
                        entropy_effective_heads=float(np.exp(-(a[a>0]*np.log(a[a>0])).sum())),zero_credit_heads=int((a==0).sum())))
    json_new(out/'complete_group_statistics.json',complete)
    json_new(out/'source_by_speed_statistics.json',source_speed)
    fields=sorted({k for r in actor_rows for k in r})
    csv_new(out/'person_equal_metrics.csv.gz',actor_rows,fieldnames=fields)
    csv_new(out/'history_error_associations.csv',correlations)
    csv_new(out/'head_imbalance_summary.csv',head_statistics)
    json_new(out/'source_macro_statistics.json',macro)
    json_new(out/'aggregation_repair.json',dict(reason='Initial aggregator inferred columns from first row; nullable direction/turn outcomes could be omitted when first row has no reliable direction',
        canonical='complete_group_statistics.json',supersedes='group_statistics.json',metrics_or_protocol_change=False,
        original_target_and_head_metrics_unchanged=True,original_files_preserved=True))


if __name__=='__main__':main()
