"""Frozen final-checkpoint comparison; reads validation predictions only."""
from collections import defaultdict
import gzip
from common import *

def load_targets(kind):
    with gzip.open(OUT/(kind+'_validation_targets.csv.gz'),'rt') as f:return list(csv.DictReader(f))

def aggregate(records):
    if not records:return dict(target_windows=0,source_persons=0,window_ADE=None,window_FDE=None,person_ADE=None,person_FDE=None)
    people=defaultdict(list)
    for r in records:people[r['source'],r['person_id']].append(r)
    return dict(target_windows=len(records),source_persons=len(people),
        window_ADE=float(np.mean([float(r['ADE']) for r in records])),window_FDE=float(np.mean([float(r['FDE']) for r in records])),
        person_ADE=float(np.mean([np.mean([float(r['ADE']) for r in v]) for v in people.values()])),
        person_FDE=float(np.mean([np.mean([float(r['FDE']) for r in v]) for v in people.values()])))

def run():
    cohorts={k:load_targets(k) for k in 'ABC'};keys=lambda r:(r['sequence'],r['frame'],r['person_id'],r['history_SHA'],r['GT_SHA'])
    assert all([keys(r) for r in cohorts[k]]==[keys(r) for r in cohorts['A']] for k in 'BC')
    verification=[]
    for k in 'ABC':
        p=CACHE/(k+'_validation_predictions.npz');pack=np.load(p);pred=pack['predictions'];gt=pack['GT'];rows=cohorts[k]
        assert pred.shape==(5203,20,12,2) and gt.shape==(5203,12,2) and np.isfinite(pred).all()
        assert all(digest(gt[i])==r['GT_SHA'] for i,r in enumerate(rows))
        distances=np.linalg.norm(pred.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1)
        ade=distances.mean(-1).min(-1);fde=distances[:,:,-1].min(-1)
        diff_a=float(np.max(np.abs(ade-np.array([float(r['ADE']) for r in rows]))));diff_f=float(np.max(np.abs(fde-np.array([float(r['FDE']) for r in rows]))))
        assert diff_a<2e-6 and diff_f<2e-6
        verification.append(dict(model=k,targets=5203,K=20,all_target_GT_SHA_match=True,independent_float64_ADE_max_difference=diff_a,independent_float64_FDE_max_difference=diff_f))
    csv_new(OUT/'independent_metric_verification.csv',verification)
    summary=[];sources=[];groups=[];paired=[]
    criteria={
        'ambiguous':lambda r:r['ambiguity']=='True',
        'non_ambiguous':lambda r:r['ambiguity']=='False',
        'no_external_neighbors':lambda r:int(r['external_neighbors'])==0,
        'one_external_neighbor':lambda r:int(r['external_neighbors'])==1,
        'multiple_external_neighbors':lambda r:int(r['external_neighbors'])>=2}
    for k,rows in cohorts.items():
        summary.append(dict(model=k,**aggregate(rows)))
        for name in sorted({r['source'] for r in rows}):sources.append(dict(model=k,source=name,**aggregate([r for r in rows if r['source']==name])))
        for name,criterion in criteria.items():groups.append(dict(model=k,group=name,**aggregate([r for r in rows if criterion(r)])))
    csv_new(OUT/'overall_metrics.csv',summary);csv_new(OUT/'source_metrics.csv',sources);csv_new(OUT/'group_metrics.csv',groups)
    for i,a in enumerate(cohorts['A']):
        b=cohorts['B'][i];c=cohorts['C'][i]
        paired.append(dict(sequence=a['sequence'],source=a['source'],frame=a['frame'],person_id=a['person_id'],history_SHA=a['history_SHA'],GT_SHA=a['GT_SHA'],ambiguity=a['ambiguity'],external_neighbors=a['external_neighbors'],
            **{f'{k}_{metric}':float(cohorts[k][i][metric]) for k in 'ABC' for metric in ['ADE','FDE']},
            B_minus_A_ADE=float(b['ADE'])-float(a['ADE']),B_minus_A_FDE=float(b['FDE'])-float(a['FDE']),C_minus_B_ADE=float(c['ADE'])-float(b['ADE']),C_minus_B_FDE=float(c['FDE'])-float(b['FDE'])))
    csv_new(OUT/'paired_target_comparison.csv.gz',paired)
    people=defaultdict(list)
    for r in paired:people[r['source'],r['person_id']].append(r)
    csv_new(OUT/'person_paired_comparison.csv',[dict(source=s,person_id=i,target_windows=len(v),**{key:float(np.mean([r[key] for r in v])) for key in paired[0] if key.endswith(('_ADE','_FDE'))}) for (s,i),v in sorted(people.items())])
    a,b,c=summary
    gates={}
    for w in ['window','person']:
        threshold=max(.005,.01*b[w+'_ADE']);fde_tol=max(.005,.01*b[w+'_FDE'])
        gates[w+'_ADE_threshold']=b[w+'_ADE']-c[w+'_ADE']>=threshold
        gates[w+'_FDE_tolerance']=c[w+'_FDE']-b[w+'_FDE']<=fde_tol
    improved_sources=[r['source'] for r in sources if r['model']=='C' and r['window_ADE']<next(x for x in sources if x['model']=='B' and x['source']==r['source'])['window_ADE']]
    gates['at_least_two_sources']=len(improved_sources)>=2
    gb={r['group']:r for r in groups if r['model']=='B'};gc={r['group']:r for r in groups if r['model']=='C'}
    gates['ambiguity_ADE_improves']=gc['ambiguous']['window_ADE']<gb['ambiguous']['window_ADE']
    severe=[]
    for group in gb:
        if gb[group]['target_windows']==0:continue
        for metric in ['window_ADE','window_FDE','person_ADE','person_FDE']:
            if gc[group][metric]-gb[group][metric]>max(.02,.05*gb[group][metric]):severe.append(dict(group=group,metric=metric,delta=gc[group][metric]-gb[group][metric]))
    gates['no_severe_group_degradation']=not severe
    gates['capacity_matched']=True;gates['cohort_GT_paired']=True;gates['GPU_under12hours']=used()<43200
    passed=all(gates.values())
    decision=dict(decision='GO: exploratory candidate only' if passed else 'NO-GO: stop explicit cross-statistic hypothesis',gates=gates,improved_sources=improved_sources,severe_group_degradation=severe,
        comparison_order=['B vs A','C vs B'],new_models_or_seeds_authorized=False,test_errors_used=False,single_seed_development_only=True,
        interpretation='Equal nominal added weights does not equal function class; single HOTEL fold/overlapping windows not independent confirmation; standard Re performance reproduction not claimed.')
    json_new(OUT/'final_decision.json',decision)
    print(json.dumps(dict(summary=summary,decision=decision),ensure_ascii=False,indent=2))

if __name__=='__main__':run()
