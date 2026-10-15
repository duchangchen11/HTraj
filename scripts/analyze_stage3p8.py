"""Paired, recording-aware descriptive analysis; no fitting or gate search."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,subprocess
from collections import defaultdict
import numpy as np
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage3p8')
REFS=['PedBase-M0','PedBase-BestCompatible','CVResidual-BestCompatible']


def read(name): return list(csv.DictReader((ROOT/name).open()))
def key(r): return r['recording'],r['sample_id'],int(r['target_id'])
def flag(s): return s in ('True','true','1','1.0')


def average_ranks(x):
    order=np.argsort(x,kind='stable'); ranks=np.empty(len(x),float); start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        ranks[order[start:end]]=(start+end-1)/2;start=end
    return ranks


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None,None
    return float(np.corrcoef(x,y)[0,1]),float(np.corrcoef(average_ranks(x),average_ranks(y))[0,1])


def main():
    p=json.loads(Path('configs/stage3p8/protocol.json').read_text()); history=read('history_features.csv'); errors=read('target_errors_long.csv')
    h={key(r):r for r in history}; predictions=defaultdict(dict)
    for r in errors:predictions[key(r)][r['model']]=r
    assert set(h)==set(predictions)
    models=p['models']+REFS
    groups=[g for g in history[0] if history[0][g] in ('True','False')]
    assert len(groups)==20
    paired=[]
    for k,f in h.items():
        row=dict(f);base=predictions[k]['CV-Last2']
        for name in models:
            e=predictions[k].get(name)
            for metric in ('ADE_m','FDE_m'):
                row[name+'__'+metric]=float(e[metric]) if e else None
                row[name+'__delta_'+metric+'_vs_Last2']=float(e[metric])-float(base[metric]) if e else None
        paired.append(row)
    csv_new(ROOT/'paired_target_results.csv',paired)
    person=defaultdict(list)
    for r in errors:person[r['recording'],r['model'],int(r['target_id'])].append(r)
    people=[]
    for (rec,model,ped),rs in person.items():
        da=[float(r['ADE_m'])-float(predictions[key(r)]['CV-Last2']['ADE_m']) for r in rs]
        df=[float(r['FDE_m'])-float(predictions[key(r)]['CV-Last2']['FDE_m']) for r in rs]
        people.append(dict(recording=rec,model=model,target_id=ped,target_windows=len(rs),ADE_m=float(np.mean([float(r['ADE_m']) for r in rs])),
            FDE_m=float(np.mean([float(r['FDE_m']) for r in rs])),delta_ADE_vs_Last2_m=float(np.mean(da)),delta_FDE_vs_Last2_m=float(np.mean(df))))
    csv_new(ROOT/'per_person_paired_results.csv',people)
    metrics=[r for r in read('recording_metrics.csv') if r['status']!='not_available']; by_metric={(r['recording'],r['model']):r for r in metrics}
    aggregates=[];matched=[]
    def aggregate(recs,names,label):
        output=[]
        for model in names:
            rs=[by_metric[rec,model] for rec in recs]
            n=sum(int(r['targets']) for r in rs)
            persons=[x for x in people if x['recording'] in recs and x['model']==model]
            for weighting in ('recording_macro','target_window_weighted','person_macro_within_recording_then_recording_macro'):
                row=dict(coverage=label,recordings='|'.join(recs),recording_count=len(recs),model=model,weighting=weighting,target_windows=n,
                    recording_person_units=len(persons))
                for metric in ('ADE_m','FDE_m'):
                    if weighting=='recording_macro':value=np.mean([float(r[metric]) for r in rs])
                    elif weighting=='target_window_weighted':value=sum(float(r[metric])*int(r['targets']) for r in rs)/n
                    else:value=np.mean([np.mean([float(x[metric]) for x in persons if x['recording']==rec]) for rec in recs])
                    row[metric]=float(value)
                output.append(row)
        return output
    recs=[f['recording'] for f in p['folds']]
    aggregates=aggregate(recs,p['models'],'all_five_fixed_recordings')
    for label in ('PedBase-M0','CVResidual-BestCompatible'):
        available=[rec for rec in recs if (rec,label) in by_metric]
        eligible=[m for m in models if all((rec,m) in by_metric for rec in available)]
        matched+=aggregate(available,eligible,'exact_shared_coverage_'+label)
    csv_new(ROOT/'aggregate_metrics.csv',aggregates);csv_new(ROOT/'reference_matched_aggregates.csv',matched)
    group_results=[]
    for r in read('group_results.csv'):
        rec,name,g=r['recording'],r['model'],r['group']
        ks=[k for k in h if k[0]==rec and flag(h[k][g]) and name in predictions[k]]
        assert len(ks)==int(r['targets'])
        da=[float(predictions[k][name]['ADE_m'])-float(predictions[k]['CV-Last2']['ADE_m']) for k in ks]
        df=[float(predictions[k][name]['FDE_m'])-float(predictions[k]['CV-Last2']['FDE_m']) for k in ks]
        group_results.append(dict(r,unique_pedestrians=len({k[2] for k in ks}),
            delta_ADE_vs_Last2_m=float(np.mean(da)) if da else None,delta_FDE_vs_Last2_m=float(np.mean(df)) if df else None,
            improved_window_fraction_ADE=float(np.mean(np.asarray(da)<-1e-12)) if da else None,
            interpretation='paired same target windows; descriptive fractions, no iid-window test'))
    csv_new(ROOT/'group_paired_comparison.csv',group_results)
    features=['last_speed_mps','mean_velocity_speed_mps','mean_scalar_speed_mps','velocity_variance_m2ps2','adjacent_velocity_change_mps',
        'net_displacement_m','total_path_m','net_path_ratio','last_mean_angle_deg','maximum_turn_angle_deg','linear_fit_RMS_m',
        'terminal_mean_velocity_difference_mps','history_frames']
    associations=[]
    for rec in recs:
        for feature in features:
            keys=[k for k in h if k[0]==rec]
            if feature=='last_mean_angle_deg':keys=[k for k in keys if float(h[k]['last_mean_angle_known'])==1]
            if feature=='net_path_ratio':keys=[k for k in keys if float(h[k]['net_path_ratio_defined'])==1]
            for model in p['models']:
                for outcome in ('ADE_m','FDE_m','delta_ADE_vs_Last2_m'):
                    x=[float(h[k][feature]) for k in keys]
                    y=[float(predictions[k][model]['ADE_m'])-float(predictions[k]['CV-Last2']['ADE_m']) if outcome.startswith('delta') else float(predictions[k][model][outcome]) for k in keys]
                    for unit in ('target_window_descriptive','person_mean_descriptive'):
                        xx,yy=x,y
                        if unit.startswith('person'):
                            person_values=defaultdict(list)
                            for k,a,b in zip(keys,x,y):person_values[k[2]].append((a,b))
                            xy=[np.mean(v,axis=0) for v in person_values.values()];xx=[v[0] for v in xy];yy=[v[1] for v in xy]
                        pearson,spearman=correlation(xx,yy)
                        associations.append(dict(recording=rec,feature=feature,model=model,outcome=outcome,unit=unit,observations=len(xx),
                            Pearson_r=pearson,Spearman_rho=spearman,p_value=None,interpretation='post-evaluation association only; correlated windows/people and small recording count; no causal or gate claim'))
    csv_new(ROOT/'feature_error_associations.csv',associations)
    examples=[]
    for rec in recs:
        for g in ('history_quasi_stationary','history_out_and_back','history_direction_changing','history_terminal_unstable'):
            keys=sorted(k for k in h if k[0]==rec and flag(h[k][g]))[:3]
            for k in keys:
                row=next(r for r in paired if key(r)==k)
                examples.append(dict(example_group=g,selection='first3 sorted historical group members, not selected by future error',**row))
    csv_new(ROOT/'fixed_history_case_examples.csv',examples)
    base_macro=next(r for r in aggregates if r['model']=='CV-Last2' and r['weighting']=='recording_macro')
    decisions=[];candidates=[]
    for model in p['models'][1:]:
        passes=[];per_record=[]
        for rec in recs:
            r,b=by_metric[rec,model],by_metric[rec,'CV-Last2']
            da=float(r['ADE_m'])-float(b['ADE_m']);df=float(r['FDE_m'])-float(b['FDE_m'])
            pass_=da<=-max(.005,.005*float(b['ADE_m'])) and df<=max(.005,.01*float(b['FDE_m']))
            passes.append(pass_);strong=[]
            for ref in REFS:
                if (rec,ref) in by_metric:
                    other=by_metric[rec,ref]
                    strong.append(dict(reference=ref,delta_ADE_m=float(r['ADE_m'])-float(other['ADE_m']),delta_FDE_m=float(r['FDE_m'])-float(other['FDE_m'])))
            per_record.append(dict(recording=rec,delta_ADE_vs_Last2_m=da,delta_FDE_vs_Last2_m=df,pass_fixed_rule=pass_,stronger_references=strong))
        macro=next(r for r in aggregates if r['model']==model and r['weighting']=='recording_macro')
        qualified=sum(passes)>=3 and macro['ADE_m']<base_macro['ADE_m'] and macro['FDE_m']<=base_macro['FDE_m']
        if qualified:candidates.append(model)
        decisions.append(dict(model=model,recording_passes=sum(passes),raw_ADE_improved_recordings=sum(r['delta_ADE_vs_Last2_m']<0 for r in per_record),
            candidate_analytic_prior=qualified,macro_delta_ADE_m=macro['ADE_m']-base_macro['ADE_m'],macro_delta_FDE_m=macro['FDE_m']-base_macro['FDE_m'],recordings=per_record))
    write_json_new(ROOT/'research_decision.json',dict(rule=p['candidate_rule'],analytic_prior_candidates=candidates,models=decisions,
        neural_coverage_complete=False,five_recording_neural_superiority_established=False,
        stop_adding_estimators=not candidates,learned_gate_supported_by_direct_experiment=False,new_training=False,ETH_evaluated=False))
    write_json_new(ROOT/'analysis_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256(__file__),inputs=[dict(path=str(ROOT/f),sha256=sha256(ROOT/f)) for f in ('history_features.csv','target_errors_long.csv','recording_metrics.csv','group_results.csv')],
        paired_targets=len(paired),person_rows=len(people),group_rows=len(group_results),association_rows=len(associations),
        threshold_tuning=False,learned_gate=False,new_neural_training=False,ETH_evaluated=False))
    print(json.dumps(dict(paired_targets=len(paired),analytic_prior_candidates=candidates,new_training=False)))


if __name__=='__main__':main()
