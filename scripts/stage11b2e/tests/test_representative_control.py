"""Pre-GPU engineering gates A-M, including independent brute-force medoids."""
import ast
import inspect
import json
import numpy as np
import pandas as pd
import pytest
import torch
from scripts.stage11b2e.common import ROOT,OUT,CACHE,CONF,Budget,sha
from scripts.stage11b2e.representative import assign,squared_medoid,select,project_record

def test_assignment_geometry_exact_ties_smaller_center_and_chunks():
    p=np.array([[0,0],[1,0],[5,0],[2,3]],dtype=float);c=np.array([[0,0],[2,0],[6,0]],dtype=float)
    for chunk in [1,3,512]:assert assign(p,c,chunk).tolist()==[0,0,2,1]

@pytest.mark.parametrize('p',[np.array([[0,0],[1,2],[3,1],[8,0]],dtype=float),
    np.array([[0,0],[0,0],[0,0],[10,0]],dtype=float),np.array([[0,0],[2,0]],dtype=float)])
def test_squared_medoid_matches_independent_small_bruteforce(p):
    idx=np.arange(len(p));j,_=squared_medoid(p,idx)
    objective=np.array([sum(float(np.dot(q-c,q-c)) for q in p) for c in p])
    assert j==int(np.flatnonzero(objective==objective.min()).min())

def test_duplicate_draw_counts_not_unweighted_unique_medoids():
    p=np.array([[0.,0.]]*8+[[10.,0.]]+[[20.,0.]])
    j,mean=squared_medoid(p,np.arange(len(p)))
    assert j==0 and mean[0]==3
    unique=np.unique(p,axis=0);u,_=squared_medoid(unique,np.arange(3));assert unique[u,0]==10

def test_tie_break_original_index_not_local_storage_order():
    p=np.array([[0.,0.],[2.,0.]])
    assert squared_medoid(p,[1,0])[0]==0

def fixture():
    p=np.tile(np.array([[0.,0.],[4.,0.],[8.,0.],[12.,0.]],dtype=np.float32),(2500,1))
    c=np.array([[float(k)*100,0] for k in range(19)],dtype=np.float32)
    return p,c,np.array([1.,2.],dtype=np.float32)

def test_empty_cell_keep_center_head0_and_exact_k20():
    P,C,S=fixture();g,a,rows,_=select(P,C,S)
    assert g.shape==(20,2) and g.dtype==np.float32
    np.testing.assert_array_equal(g[0],S)
    for r in rows:
        if r['empty_fallback']:np.testing.assert_array_equal(g[r['head']],C[r['center_index']]);assert r['proposal_index']==-1
    assert sum(r['population'] for r in rows)==10000

def test_repeat_output_determinism_and_no_random_state_change():
    P,C,S=fixture();before=np.random.get_state()
    a=select(P,C,S)[0];b=select(P,C,S)[0]
    np.testing.assert_array_equal(a,b)
    after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]

def test_future_counterfactual_projection_has_no_gt_dependency():
    P,C,S=fixture();rec={'sample_id':'synthetic','tensors':{'proposals_raw':torch.tensor(P)[None,None],
        'centers_raw':torch.tensor(C),'softargmax_raw':torch.tensor(S)},'future_GT':np.zeros((12,2))}
    view0=project_record(rec);rec['future_GT']=np.full((12,2),-1e8);view1=project_record(rec)
    assert all(np.array_equal(x,y) for x,y in zip(view0[1:],view1[1:]))
    np.testing.assert_array_equal(select(*view0[1:])[0],select(*view1[1:])[0])

def test_resize_raw_network_roundtrip_power_of_two():
    x=torch.tensor([[31.25,89.75],[1000.,2000.]],dtype=torch.float32)
    assert torch.equal(x*.25/.25,x)

def test_budget_guard_accounts_prior_and_startup():
    clock=[100.];b=Budget(prior=20,clock=lambda:clock[0],started=90.)
    assert b.elapsed()==30
    clock[0]=3359.
    with pytest.raises(TimeoutError):b.guard(30)

def test_new_cache_all998_membership_head0_and_reverse_gate():
    report=json.loads((OUT/'selection_audit.json').read_text());assert report['passed'] and report['targets']==998
    assert report['all998_reverse_bitwise'] and report['all_head0_bitwise'] and not report['actual_GT_loaded']
    p=ROOT/report['new_goals_path'];assert sha(p)==report['new_goals_file_SHA']
    pack=torch.load(p,map_location='cpu',weights_only=True);assert pack['goals'].shape==(20,998,2)
    assert len(set(pack['sample_ids']))==998 and torch.isfinite(pack['goals']).all()
    pop=pd.read_csv(OUT/'representative_selection.csv');assert len(pop)==998*19 and (pop.groupby('sample_id').population.sum()==10000).all()
    for r in report['GT_counterfactual']:assert r['selection_inputs_bitwise'] and r['representatives_bitwise']

def test_all_real_selected_indices_belong_to_correct_frozen_cells():
    m=pd.read_csv(ROOT/'results/tables/stage11b2c/tensor_manifest.csv').set_index('sample_id')
    rows=pd.read_csv(OUT/'representative_selection.csv')
    for sid in json.loads((CONF/'generation_config.json').read_text())['regression_ids']:
        pack=torch.load(ROOT/m.loc[sid,'path'],map_location='cpu',weights_only=True);_,P,C,S=project_record(pack);a=assign(P,C)
        for r in rows[rows.sample_id==sid].itertuples():
            if r.proposal_index>=0:
                assert a[r.proposal_index]==r.center_index
                np.testing.assert_array_equal(P[r.proposal_index],[r.new_x,r.new_y])

def test_original_decoder_inner_math_is_preserved():
    old=ast.parse((ROOT/'scripts/stage11b2c/adapter.py').read_text());new=ast.parse((ROOT/'scripts/stage11b2e/decoder.py').read_text())
    a=next(n for n in ast.walk(old) if isinstance(n,ast.For) and ast.unparse(n.target)=='goal')
    b=next(n for n in ast.walk(new) if isinstance(n,ast.For) and ast.unparse(n.target)=='goal')
    assert ast.unparse(a).replace('self.templates','templates')==ast.unparse(b)

def test_selector_source_has_no_future_or_sampling_api():
    from scripts.stage11b2e import representative
    source=inspect.getsource(representative.select)
    assert all(word not in source for word in ['GT','future','multinomial','cluster_proposals','kmeans','random','oracle'])
