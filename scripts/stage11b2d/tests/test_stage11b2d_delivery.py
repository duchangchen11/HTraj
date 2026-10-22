"""Acceptance of real artifacts, distinct from synthetic geometry checks."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from scripts.stage11b2d.common import ROOT,OUT,OLD,CONF,FIG,CACHE,sha

def test_literature_methods_versions_and_overlap_are_auditable():
    papers=pd.read_csv(OUT/'literature_review.csv');m=pd.read_csv(OUT/'literature_overlap_matrix.csv')
    assert len(papers)==12 and set(m.direction)==set('ABCDEF')
    assert set(['YNet','PECNet','DLow','AgentFormer','TUTR']).issubset(papers.key)
    assert {2024,2025,2026}.issubset(set(papers.year))
    assert papers.evidence_level.eq('FULLTEXT_METHOD_REVIEWED').all()
    for r in papers.itertuples():
        assert sha(CACHE/'literature'/(r.key+'.pdf'))==r.PDF_SHA
        assert sha(CACHE/'literature'/(r.key+'.txt'))==r.text_SHA
    assert 'ICPR' in papers.set_index('key').loc['LRVQ2024','venue']
    assert 'Waymo' in papers.set_index('key').loc['GMM2026','datasets']

def test_frozen_case_coordinates_and_figure_hashes():
    cases=json.loads((OUT/'frozen_case_data.json').read_text())
    assert [r['sample_id'] for r in cases]==json.loads((CONF/'analysis_protocol.json').read_text())['case_ids']
    for r in cases:
        pack=torch.load(ROOT/r['source_cache'],map_location='cpu',weights_only=True)
        idx=r['local_proposal_indices']
        np.testing.assert_array_equal(np.asarray(r['local_proposal_coordinates']),pack['tensors']['proposals_raw'][0,0].numpy()[idx])
        np.testing.assert_array_equal(np.asarray(r['center_coordinates']),pack['tensors']['centers_raw'].numpy())
    figures=json.loads((OUT/'figure_manifest.json').read_text());assert len(figures)==5
    for r in figures:
        assert sha(FIG/(r['name']+'.png'))==r['PNG_SHA']
        assert sha(FIG/(r['name']+'.svg'))==r['SVG_SHA'];ET.parse(FIG/(r['name']+'.svg'))

def test_all_required_tables_and_no_automatic_algorithm_go():
    for p in ['protocol_frozen.json','cache_audit.json','cluster_diagnostics.csv','local_proposal_assignments.csv.gz',
              'compression_failure_comparison.csv','recording_comparison.csv','threshold_support_review.csv',
              'literature_review.csv','literature_overlap_matrix.csv','hypothesis_decision.json','final_decision.json']:
        assert (OUT/p).is_file()
    d=json.loads((OUT/'final_decision.json').read_text())
    assert d['new_GPU_seconds']==d['new_training_steps']==d['new_model_inference']==0
    assert d['compression_targets_analyzed']==86 and not d['Stage11C_GO']
    h=json.loads((OUT/'hypothesis_decision.json').read_text())
    assert len(h['hypotheses'])==1 and not h['innovation_admission']
    assert not h['hypotheses'][0]['GT_for_selection']

def test_recording_weighting_uses_all_five_including_small_cells():
    r=pd.read_csv(OUT/'recording_comparison.csv');scenes=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4']
    q=r.set_index('scope').loc[scenes]
    assert q.targets.sum()==998 and q.compression_fail.sum()==86
    macro=pd.read_csv(OUT/'recording_macro_rates.csv').set_index('metric')
    assert abs(q.compression_loss_rate_all.mean()-macro.loc['compression_loss_rate_all','mean'])<1e-12
    assert abs(r.set_index('scope').loc['ALL998','compression_loss_rate_all']-86/998)<1e-12
