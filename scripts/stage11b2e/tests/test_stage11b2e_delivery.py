"""Real completed control artifacts, not extra GPU runs or parameter searches."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from scripts.stage11b2e.common import ROOT,OUT,CONF,CACHE,FIG,PP,sha
from scripts.stage11b2e.metrics import gate

def test_frozen_numerical_sources_and_protocol_remain_identical():
    f=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'experiment_protocol.json')==f['protocol_SHA']
    assert sha(CONF/'generation_config.json')==f['generation_config_SHA']
    for row in f['executable_files']:assert sha(ROOT/row['path'])==row['sha256']

def test_actual_all998_paths_and_head0_match_verified_original():
    r=json.loads((OUT/'generation_execution.json').read_text())
    assert r['passed'] and r['generated']==998 and r['parameters_unchanged']
    assert sha(ROOT/r['alternative_paths_path'])==r['alternative_paths_file_SHA']
    n=torch.load(CACHE/'alternative_paths.pt',map_location='cpu',weights_only=True)
    o=torch.load(PP,map_location='cpu',weights_only=True);idx={s:i for i,s in enumerate(o['sample_ids'])}
    assert n['paths'].shape==(20,998,12,2) and torch.isfinite(n['paths']).all()
    assert set(n['sample_ids'])==set(idx) and len(set(n['sample_ids']))==998
    for j,s in enumerate(n['sample_ids']):
        assert torch.equal(n['goals'][0,j],o['goals'][0,idx[s]])
        assert torch.equal(n['paths'][0,j],o['paths'][0,idx[s]])

def test_paired_cohort_conservation_and_head_change_mapping():
    r=pd.read_csv(OUT/'failure_recovery.csv');p=pd.read_csv(OUT/'paired_path_metrics.csv')
    g=pd.read_csv(OUT/'paired_goal_metrics.csv');h=pd.read_csv(OUT/'paired_head_metrics.csv')
    assert len(r)==len(p)==len(g)==998 and len(h)==19960 and r.sample_id.is_unique
    assert set(r.sample_id)==set(p.sample_id)==set(g.sample_id)==set(h.sample_id)
    assert (g.set_index('sample_id').changed_decoder_map_centers==p.set_index('sample_id').changed_path_heads).all()
    a=int(r.OriginalGoalCover25.sum());b=int(r.AlternativeGoalCover25.sum())
    assert b-a==int((r.paired_outcome25=='recovered').sum())-int((r.paired_outcome25=='lost').sum())
    assert sum(r.original_category25=='proposal_hit_final_miss')==86
    assert sum(r.original_category25=='zero_support_final_miss')==77
    assert h[h['head']==0].GoalBitwiseUnchanged.all() and h[h['head']==0].PathBitwiseUnchanged.all()

def test_recording_macro_and_all_four_gate_recompute_exactly():
    f=pd.read_csv(OUT/'scene_comparison.csv').set_index('scope')
    names=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4']
    for k in ['GoalCover25','minADE20','minFDE20']:
        for prefix in ['Original','Alternative']:
            assert np.isclose(f.loc[names,prefix+k].mean(),f.loc['FIVE_RECORDING_MACRO',prefix+k],rtol=0,atol=1e-12)
    get=lambda row,p:{k:f.loc[row,p+k] for k in ['GoalCover25','minADE20','minFDE20']}
    d=gate(get('FIVE_RECORDING_MACRO','Original'),get('FIVE_RECORDING_MACRO','Alternative'),get('OTHER4','Original'),get('OTHER4','Alternative'))
    saved=json.loads((OUT/'final_decision.json').read_text())
    assert d['status']==saved['status'] and d['conditions']==saved['conditions']
    assert json.loads((OUT/'gpu_budget.json').read_text())['actual_GPU_process_seconds']<3300

def test_six_real_figures_and_frozen_case_selection():
    f=json.loads((OUT/'figure_manifest.json').read_text());assert len(f)==6
    for r in f:
        assert sha(FIG/(r['name']+'.png'))==r['PNG_SHA']
        assert sha(FIG/(r['name']+'.svg'))==r['SVG_SHA']
        ET.parse(FIG/(r['name']+'.svg'))
    r=pd.read_csv(OUT/'failure_recovery.csv');c=pd.read_csv(OUT/'case_manifest.csv')
    for _,row in c.iterrows():
        q=r[r.paired_outcome25==row.outcome].sort_values('sample_id',kind='stable')
        assert row.sample_id==q.iloc[(len(q)-1)//2].sample_id
