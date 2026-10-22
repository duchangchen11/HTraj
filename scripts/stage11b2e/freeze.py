"""CPU cache admission and protocol freeze. No future labels or model calls."""
from .common import *
import pandas as pd
import numpy as np
import torch

def main():
    cpu_only();CACHE.mkdir(parents=True,exist_ok=True)
    assert json.loads((OUT/'history_before.json').read_text())['passed']
    assert sha(CP)==CK and sha(PP)==PSHA
    m=pd.read_csv(OLD/'tensor_manifest.csv');assert len(m)==m.sample_id.nunique()==998
    missing=[p for p in m.path if not (ROOT/p).is_file()]
    if missing:
        jnew(OUT/'blocked.json',dict(status='BLOCKED_CACHE_MISSING',paths=missing));raise FileNotFoundError(missing[0])
    old=torch.load(PP,map_location='cpu',weights_only=True);assert set(old)=={'sample_ids','paths','goals','checkpoint_SHA','epoch','K','TTST_proposals','units'}
    assert old['checkpoint_SHA']==CK and old['K']==20 and old['TTST_proposals']==10000
    order={sid:i for i,sid in enumerate(old['sample_ids'])};assert set(order)==set(m.sample_id)
    pop=pd.read_csv(ROOT/'results/tables/stage11b2d/cluster_diagnostics.csv',usecols=['sample_id','center_index','Population'])
    assert len(pop)==998*19 and (pop.groupby('sample_id').Population.sum()==10000).all() and (pop.Population>0).all()
    ids=pd.read_csv(ROOT/'results/tables/stage11b2d/target_diagnostics.csv',usecols=['sample_id']);assert set(ids.sample_id)==set(order)
    picks=[m.sample_id.iloc[k] for k in [0,499,997]];ref=[];refg=[];refp=[]
    for row in m.to_dict('records'):
        assert sha(ROOT/row['path'])==row['sha256'],row['sample_id']
        p=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True);t=p['tensors'];sid=row['sample_id'];i=order[sid]
        assert p['sample_id']==sid and p['checkpoint_SHA']==CK and p['resize']==.25
        for key,name in [('indices','indices_SHA'),('proposals_network','proposal_SHA'),('centers_network','centers_SHA'),('goals','goal_SHA'),('paths','path_SHA')]:assert digest(t[key])==row[name]
        assert t['indices'].shape==(1,10000) and t['indices'].dtype==torch.int64
        W=t['sigmoid'].shape[-1];index=t['indices'].numpy().ravel();xy=np.stack([index%W,index//W],axis=-1).astype(np.float32)
        assert np.array_equal(xy/.25,t['proposals_raw'][0,0].numpy())
        assert torch.equal(t['centers_network']/.25,t['centers_raw'])
        assert torch.equal(t['goals'][0,0],t['softargmax_raw']) and torch.equal(t['goals'][1:,0],t['centers_raw'])
        assert torch.equal(t['goals'],old['goals'][:,i:i+1]) and torch.equal(t['paths'],old['paths'][:,i:i+1])
        if sid in picks:ref.append(sid);refg.append(t['goals'][:,0]);refp.append(t['paths'][:,0])
    expected=CACHE/'original_regression_reference.pt';assert not expected.exists()
    torch.save(dict(sample_ids=ref,goals=torch.stack(refg,dim=1),paths=torch.stack(refp,dim=1),checkpoint_SHA=CK),expected)
    ex=json.loads((OLD/'input_export.json').read_text());hist=ex['artifacts']['validation_history_only'];assert sha(ROOT/hist['path'])==hist['sha256']
    h=torch.load(ROOT/hist['path'],map_location='cpu',weights_only=True);assert set(h)=={'sample_ids','scenes','history'} and set(h['sample_ids'])==set(order)
    gen=json.loads((OLD/'generation_execution.json').read_text());sem=gen['semantic_caches']
    for r in sem.values():assert sha(ROOT/r['path'])==r['file_SHA']
    source_files=[OLD/'tensor_manifest.csv',ROOT/'results/tables/stage11b2d/cluster_diagnostics.csv',
      ROOT/'results/tables/stage11b2d/target_diagnostics.csv',ROOT/'results/tables/stage11b1c/validation_target_manifest.csv',
      OLD/'data_provenance.json',OLD/'input_export.json',OLD/'generation_execution.json']
    cfg=dict(stage='Stage11B-2E',scope='CONTROL_EXPERIMENT_ONLY',base_commit=BASE,
      checkpoint_SHA=CK,original_prediction_SHA=PSHA,official_commit=OFFICIAL,sample_ids=m.sample_id.tolist(),
      sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in source_files],
      identity='same998 development targets/GT; five reused recordings; not independent confirmation',
      recordings=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'],
      assignment='FINAL_CENTER_VORONOI_ASSIGNMENT',assignment_formula='argmin_k ||P_j-C_k||^2 in float64; exact ties smaller center index',
      medoid_formula='argmin_p_in_Zk sum_q_in_Zk ||q-p||^2; implemented argmin ||p-mean(Zk)||^2',
      selector_precision='CPU float64; mean counts duplicate draws; copy chosen original float32 coordinate',
      representative_tie='exact equal float64 criterion -> lowest original proposal index; no tolerance',
      assignment_chunk=512,head_order='head0 original Softargmax bitwise; heads1..19 original final-cell order',
      empty_cell='keep original center; proposal index=-1; log, no random replacement',
      duplicate_goals='retain, count, never fill or reorder',radii=[10,25,50],resize=.25,
      frozen_decoder_map_rounding='original get_patch uses np.round in network pixels; retain it, no extra rounding in representative selection',
      candidate_count=20,proposal_draws=10000,unit='original SDD reference pixels',
      decoder_regression_ids=picks,regression='original20 paths bitwise for all3; stop if any mismatch',
      metrics=['ADE1','FDE1','minADE20','minFDE20','minGoalFDE20','GoalFDE1','GoalCover10','GoalCover25','GoalCover50',
        'GoalPairwiseDispersion','PathPairwiseDispersion','UniqueGoals','UniquePaths','Distortion19','Distortion20','FixedCellDistortion19','probability_mass_union10/25/50'],
      aggregation=['all998 window weighted','five recordings equal weight','each recording','deathCircle','other4 pooled','all frozen Stage2A history groups'],
      frozen_gate={'macro_GoalCover25_gain_min':.01,'macro_minADE20_relative_drop_min':.01,
        'macro_minFDE20_relative_degradation_max':.01,'other4_GoalCover25_gain':'strictly >0','all_four_required':True},
      success='CONTROL_GO only, never novelty GO',failure='CONTROL_NO_GO if any gate fails; no alternative formula/seed/radius/checkpoint search',
      case_rule='For every paired25px outcome recovered/lost/stayed_fail/stayed_cover: sort sample_id and choose floor((n-1)/2); absent outcomes explicitly absent; use same original/alternative paths',
      plotting={'backend':'python','claim':'show paired control outcome whether positive or negative','archetype':'quantitative grid plus real spatial case panels',
        'formats':['PNG','SVG'],'data':'actual same-target predictions; GT offline only','statistics':'no overlapping-window independent p-values'},
      gpu_hard_seconds=3600,gpu_safety_seconds=3300,parent_meter='whole process including startup/load/regression/decode/write/exit',
      restrictions=['no training','no new TTST draws','no KMeans rerun','no changed threshold','no GT in generation','no sealed test','no second method'])
    jnew(CONF/'experiment_protocol.json',cfg)
    g=dict(checkpoint_path=str(CP.relative_to(ROOT)),checkpoint_SHA=CK,original_state_SHA=gen['model_state_SHA'],
      history=hist,semantics=sem,regression_reference=dict(path=str(expected.relative_to(ROOT)),sha256=sha(expected)),
      regression_ids=picks,experiment_protocol_SHA=sha(CONF/'experiment_protocol.json'),budget={'hard':3600,'safe':3300},
      input_schema=['sample_id','history8','frozen_semantics','frozen_weights','alternative_goals'],GT_fields=False)
    jnew(CONF/'generation_config.json',g)
    sources=sorted((ROOT/'scripts/stage11b2e').glob('*.py'))
    sources += [ROOT/p for p in ['scripts/stage11b0/runtime.py','scripts/stage11b0/training_adapter.py','scripts/stage11b1/common.py','scripts/stage11b2c/adapter.py']]
    jnew(OUT/'protocol_frozen.json',dict(protocol_SHA=sha(CONF/'experiment_protocol.json'),generation_config_SHA=sha(CONF/'generation_config.json'),
      executable_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sources],pre_freeze_commit=commit(),required_freeze_commit_before_execution=True))
    jnew(OUT/'cache_audit.json',dict(passed=True,targets=998,all_proposal_order_coordinate_and_SHA=True,all19centers_softargmax_20goals_paths_bitwise=True,
      checkpoint_SHA=CK,original_prediction_SHA=PSHA,all_source_populations_nonempty=True,GT_not_loaded=True,CUDA_initialized=False,
      expected_regression_ids=ref,original_goals_shape=list(old['goals'].shape),original_paths_shape=list(old['paths'].shape)))
    print('Cache admission998 PASS; freeze protocol/code commit before selection',flush=True)

if __name__=='__main__':main()
