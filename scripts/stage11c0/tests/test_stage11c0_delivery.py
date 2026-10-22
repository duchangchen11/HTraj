"""Real artifact admission after all groups have actually completed."""
import json
import xml.etree.ElementTree as ET
import numpy as np,pandas as pd,torch
from scripts.stage11c0.common import ROOT,CONF,OUT,FIG,checked,sha,parent_seconds

def test_c0_frozen_math_and_configuration_unchanged_after_generation():
    f=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'experiment_protocol.json')==f['protocol_SHA'] and sha(CONF/'generation_config.json')==f['generation_config_SHA']
    for r in f['executable_files']:assert sha(ROOT/r['path'])==r['sha256']
    old=json.loads((OUT/'protocol_frozen_failed_attempt1.json').read_text())
    assert old['protocol_SHA']==f['protocol_SHA'] and old['generation_config_SHA']==f['generation_config_SHA']
    before={r['path']:r['sha256'] for r in old['executable_files']};after={r['path']:r['sha256'] for r in f['executable_files']}
    assert {p for p in before if before[p]!=after[p]}=={'scripts/stage11c0/worker.py'}

def test_c0_all_six_groups_real_shapes_and_fixed_goals_identity():
    cfg=json.loads((CONF/'generation_config.json').read_text());old=checked(cfg['original_predictions']);hp=checked(cfg['history'])
    for kind,name in [('full','full_generation.json'),('fixed','fixed_generation.json')]:
        log=json.loads((OUT/name).read_text());assert log['passed'] and log['generated']==2994 and not log['GT_loaded'] and log['parameters_unchanged']
        assert len(log['groups'])==3
        for r in log['groups']:
            x=checked(r);assert x['sample_ids']==hp['sample_ids']==old['sample_ids']
            assert x['paths'].shape==(20,998,12,2) and x['goals'].shape==(20,998,2)
            assert torch.isfinite(x['paths']).all() and torch.isfinite(x['goals']).all()
            if kind=='fixed':assert torch.equal(x['goals'],old['goals'])

def test_c0_actual_fixed_path_displacements_match_saved_arrays():
    cfg=json.loads((CONF/'generation_config.json').read_text());old=checked(cfg['original_predictions'])
    frame=pd.read_csv(OUT/'fixed_goal_path_metrics.csv').set_index(['condition','sample_id'])
    for r in json.loads((OUT/'fixed_generation.json').read_text())['groups']:
        x=checked(r);c=r['condition'][:2]
        delta=(x['paths'].double()-old['paths'].double()).norm(dim=-1)
        assert np.allclose(delta.mean(dim=(0,2)).numpy(),[frame.loc[(c,s),'Path_point_mean_displacement'] for s in x['sample_ids']],rtol=0,atol=1e-9)
        count=(x['paths']!=old['paths']).any(dim=-1).any(dim=-1).sum(dim=0).numpy()
        assert np.array_equal(count,[frame.loc[(c,s),'Path_changed_heads'] for s in x['sample_ids']])

def test_c0_original164_IDs_and_transition_conservation():
    f=pd.read_csv(OUT/'a25_transition.csv');original=pd.read_csv(ROOT/'results/tables/stage11b2d/target_diagnostics.csv')
    oldfail=set(original[original.category25!='both_hit'].sample_id);assert len(oldfail)==164
    for c,q in f.groupby('condition'):
        assert len(q)==998 and q.sample_id.is_unique and set(q[q.original_A25_failure].sample_id)==oldfail
        assert q.OriginalGoalCover25.sum()==834
        assert q.AlternativeGoalCover25.sum()-834==(q.outcome=='recovered').sum()-(q.outcome=='lost').sum()

def test_c0_equal_recording_macro_and_empty_group_values():
    f=pd.read_csv(OUT/'scene_comparison.csv');names=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4']
    for (exp,c),q in f.groupby(['experiment','condition']):
        source=q[q.scope.isin(names)];macro=q[q.scope=='FIVE_RECORDING_MACRO'].iloc[0]
        assert len(source)==5
        for key in ['ADE1','FDE1','minADE20','minFDE20','GoalCover25']:assert abs(source[key].mean()-macro[key])<1e-9
    groups=pd.read_csv(OUT/'motion_group_comparison.csv');zero=groups[groups.targets==0]
    if len(zero):assert zero.minADE20.isna().all() and zero.minFDE20.isna().all()

def test_c0_original_regression_and_failure_wall_budget_preserved():
    r=json.loads((OUT/'baseline_regression.json').read_text());assert r['passed'] and r['full_original_prediction_bitwise'] and r['real_targets']==3 and r['repeats']==2
    v=json.loads((OUT/'fixed_repeat_reverse.json').read_text());assert v['passed'] and v['comparisons']==18 and v['all20paths_repeat_reverse_bitwise']
    fail=json.loads((OUT/'engineering_failure_and_fix.json').read_text())
    for r in fail['preserved']:
        assert sha(ROOT/r['preserved_path'])==r['sha256']
    budget=json.loads((OUT/'gpu_budget.json').read_text());assert abs(budget['actual_GPU_process_seconds']-parent_seconds())<1e-9
    assert budget['actual_GPU_process_seconds']<6600
    assert any(r['returncode']==1 and abs(r['wall_seconds']-fail['failed_GPU_process_seconds'])<1e-9 for r in budget['processes'])

def test_c0_all_ten_actual_figures_and_fixed_cases():
    figs=json.loads((OUT/'figure_manifest.json').read_text());assert len(figs)==10
    for row in figs:
        assert sha(FIG/(row['name']+'.png'))==row['PNG_SHA'] and sha(FIG/(row['name']+'.svg'))==row['SVG_SHA']
        ET.parse(FIG/(row['name']+'.svg'))
    q=pd.read_csv(OUT/'a25_transition.csv');cases=pd.read_csv(OUT/'qualitative_case_manifest.csv')
    for _,row in cases[cases.condition!='ALL'].iterrows():
        ids=sorted(q[(q.condition==row.condition)&(q.outcome==row.case_type)].sample_id)
        assert row.sample_id==ids[(len(ids)-1)//2]
