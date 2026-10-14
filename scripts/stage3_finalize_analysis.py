"""Frozen Stage3 endpoints, preregistered selection and auxiliary interventions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,subprocess
import numpy as np
import torch
from src.training.order_verification import datasets
from src.training.social_diagnostics import csv_new
from src.data.preprocessing import sha256,write_json_new
from src.evaluation.relation_routing import cache_validation,evaluate
from src.models.relation_routing import build_routing_model
from src.models.order_verification import build_order_model

ROOT=Path('results/tables/stage3')


def selection(p,new,frozen):
    refs={r['model']:r for r in frozen if int(r['seed'])==17}
    models={r['model']:r for r in new if int(r['seed'])==17}
    decisions=[]
    base_groups={r['group']:r for r in csv.DictReader((ROOT/'frozen/PedBase/seed_17/groups_best.csv').open())}
    no_update=models['Routing-NoUpdate'];qualified=[]
    for name in ['Receiver-Fix','Relation-Routing']:
        r=models[name];g={s['group']:s for s in csv.DictReader((ROOT/'new'/name/'seed_17/groups_best.csv').open())}
        primary=all(float(r['best_ADE_m'])<=.99*float(ref['best_ADE_m']) and float(r['best_FDE_m'])<=1.01*float(ref['best_FDE_m']) for ref in refs.values())
        safe_full=float(g['history_full8']['ADE_m'])<=1.01*float(base_groups['history_full8']['ADE_m'])
        targeted=int(g['one_frame_with_neighbors']['ade_count'])>=10 and float(g['one_frame_with_neighbors']['ADE_m'])<=.90*float(base_groups['one_frame_with_neighbors']['ADE_m'])
        safe_global=all(float(r['best_ADE_m'])<=1.01*float(ref['best_ADE_m']) and float(r['best_FDE_m'])<=1.01*float(ref['best_FDE_m']) for ref in refs.values())
        memory=name!='Relation-Routing' or float(r['best_ADE_m'])<=.995*float(no_update['best_ADE_m'])
        passed=(primary or (targeted and safe_global)) and safe_full and memory
        decisions.append(dict(model=name,primary_ADE_1pct=primary,full8_safe=safe_full,targeted_ADE_10pct=targeted,overall_safe=safe_global,memory_vs_NoUpdate_0p5pct=memory,qualified=passed))
        if passed:qualified.append(r)
    selected=[r['model'] for r in sorted(qualified,key=lambda x:(float(x['best_ADE_m']),float(x['best_FDE_m'])))[:2]]
    write_json_new(ROOT/'round2_selection.json',dict(selected_models=selected,decisions=decisions,first_round_complete=True,
        candidate_universe=['Receiver-Fix','Relation-Routing'],rule_from_frozen_protocol_sha256=sha256('configs/stage3/protocol.json'),
        rationale='Primary candidate models only; other models are attribution controls, not additional searched architectures',
        seeds_if_selected=[17,42,2026],seed17_reused_identical_first_run=True,test_evaluated=False,
        confirmation_evaluation='not possible: zero time-disjoint validation windows',analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--select-only',action='store_true');args=parser.parse_args()
    p=json.loads(Path('configs/stage3/protocol.json').read_text());new=[]
    for name in p['new_models']:
        s=json.loads((ROOT/'new'/name/'seed_17/summary.json').read_text());assert s['status']=='completed' and s['last_epoch']==30
    for path in sorted((ROOT/'new').glob('*/seed_*/summary.json')):new.append(json.loads(path.read_text()))
    frozen=list(csv.DictReader((ROOT/'frozen_baselines.csv').open()))
    if not (ROOT/'round2_selection.json').exists():selection(p,new,frozen)
    if args.select_only:return
    csv_new(ROOT/'new_runs.csv',new);all_rows=[*new,*frozen];csv_new(ROOT/'all_train_runs.csv',all_rows)
    seed17=[r for r in all_rows if int(r['seed'])==17]
    csv_new(ROOT/'seed17_comparison.csv',[dict(model=r['model'],parameter_count=r['parameter_count'],best_epoch=r['best_epoch'],
        best_ADE_m=r['best_ADE_m'],paired_FDE_m=r['best_FDE_m'],last_ADE_m=r['last_ADE_m'],last_FDE_m=r['last_FDE_m'],
        training_seconds=r['actual_training_seconds'],trained_in_stage3=r.get('trained_in_stage3',False) in (True,'True'),source_commit=r['source_commit']) for r in seed17])
    groups=[];diagnostics=[]
    for r in all_rows:
        name,seed=r['model'],int(r['seed']);sub='new' if r.get('trained_in_stage3',False) in (True,'True') else 'frozen';run=ROOT/sub/name/f'seed_{seed}'
        for endpoint in ('best','last'):
            for g in csv.DictReader((run/f'groups_{endpoint}.csv').open()):groups.append(dict(model=name,seed=seed,endpoint=endpoint,**g))
            diagnostics.append(dict(model=name,seed=seed,endpoint=endpoint,**json.loads((run/f'diagnostics_{endpoint}.json').read_text())))
    for g in csv.DictReader((ROOT/'CV/groups.csv').open()):groups.append(dict(model='CV',seed=0,endpoint='analytical',**g))
    refs={(g['model'],g['seed'],g['endpoint'],g['group']):g for g in groups if g['model'] in ('PedBase','Dynamic-GNN-RN')}
    for g in groups:
        seed=17 if g['model']=='CV' else g['seed'];endpoint='best' if g['model']=='CV' else g['endpoint']
        for ref in ('PedBase','Dynamic-GNN-RN'):
            b=refs.get((ref,seed,endpoint,g['group']))
            for metric in ('ADE_m','FDE_m'):
                g['delta_'+ref+'_'+metric]=float(g[metric])-float(b[metric]) if b and g[metric] and b[metric] else None
    csv_new(ROOT/'all_groups.csv',groups);csv_new(ROOT/'all_diagnostics.csv',diagnostics)
    torch.set_num_threads(1);_,val=datasets(p);batches=cache_validation(val)
    # Re-evaluate old H1 with the Stage3 grouping, never retrain/overwrite it.
    old_summary=json.loads(Path('results/tables/stage2p6/new/TOHI-H1/seed_17/summary.json').read_text());out=ROOT/'supplementary_H1';out.mkdir()
    for endpoint in ('best','last'):
        path=Path('experiments/stage2p6/new/TOHI-H1/seed_17/checkpoints')/(endpoint+'.pt');assert sha256(path)==old_summary[endpoint+'_checkpoint_sha256']
        model=build_order_model('TOHI-H1');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);r=evaluate(model,batches,torch.device('cpu'))
        assert abs(r['ADE_m']-old_summary[endpoint+'_ADE_m'])<1e-6 and abs(r['FDE_m']-old_summary[endpoint+'_FDE_m'])<1e-6
        csv_new(out/f'groups_{endpoint}.csv',r['groups'])
    write_json_new(out/'audit.json',dict(origin_summary='results/tables/stage2p6/new/TOHI-H1/seed_17/summary.json',old_summary=old_summary,
        source_files_unchanged=True,protocol_identical_for_common_training_fields=True,purpose='supplementary receiver-policy single-order comparison; not new training',test_evaluated=False))
    interventions=[]
    for name in ['Receiver-Fix','Relation-Routing']:
        path=Path('experiments/stage3/new')/name/'seed_17/checkpoints/best.pt';model=build_routing_model(name)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);ref=evaluate(model,batches,torch.device('cpu'))
        for case,options in [('full',None),('no_messages',dict(remove_messages=True)),('uniform',dict(uniform=True)),('reset_relation_each_frame',dict(no_update=True)),('scene_random',dict(random_graph=True))]:
            r=evaluate(model,batches,torch.device('cpu'),intervention=options,reference=ref['predictions'])
            interventions.append(dict(model=name,seed=17,case=case,checkpoint_sha256=sha256(path),ADE_m=r['ADE_m'],FDE_m=r['FDE_m'],
                delta_ADE_m=r['ADE_m']-ref['ADE_m'],delta_FDE_m=r['FDE_m']-ref['FDE_m'],**r['diagnostics'],
                independent_training=False,test_evaluated=False))
    csv_new(ROOT/'fixed_checkpoint_interventions.csv',interventions)
    write_json_new(ROOT/'analysis_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        data_manifest_sha256=sha256(p['data_manifest']),subset_manifest_sha256=sha256(p['subset_manifest']),test_evaluated=False,
        first_round_training_runs=6,first_round_epochs=180,first_round_optimizer_steps=23040,
        confirmation_evaluated=False,confirmation_unavailable=True))


if __name__=='__main__':main()
