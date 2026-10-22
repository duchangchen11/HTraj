"""Verify existing caches/identities and freeze definitions before new statistics."""
from .common import *
import sys
import time
import pandas as pd
import numpy as np
import torch

def protect(freeze):
    path = CONF / 'protected_files.csv.gz'
    if freeze:
        with gzip.open(ROOT / 'configs/stage11b2c/protected_files.csv.gz', 'rt') as f:
            records = {r['path']: r['sha256'] for r in csv.DictReader(f)}
        names = subprocess.check_output(['git','ls-tree','-r','--name-only',BASE], cwd=ROOT, text=True).splitlines()
        names += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/stage11b2c/cache').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
        for name in names:
            s = sha(ROOT/name)
            assert name not in records or records[name] == s, name
            records[name] = s
        csvnew(path, [dict(path=k, sha256=v) for k,v in sorted(records.items())])
    with gzip.open(path, 'rt') as f:
        records = list(csv.DictReader(f))
    bad = [r['path'] for r in records if not (ROOT/r['path']).is_file() or sha(ROOT/r['path']) != r['sha256']]
    jnew(OUT/('history_before.json' if freeze else 'history_after.json'),
         dict(passed=not bad, files=len(records), changed=bad, base=BASE))
    assert not bad, bad[:10]
    print('Historical SHA protection PASS', len(records), flush=True)

def main():
    cpu_guard()
    if '--after' in sys.argv:
        protect(False)
        return
    manifest_check=pd.read_csv(OLD/'tensor_manifest.csv')
    missing=[r for r in manifest_check.path if not (ROOT/r).is_file()]
    if missing:
        jnew(OUT/'blocked.json',dict(status='BLOCKED_CACHE_MISSING',missing=missing))
        raise FileNotFoundError(missing[0])
    protect(True)
    start=time.perf_counter()
    m=pd.read_csv(OLD/'tensor_manifest.csv')
    s=pd.read_csv(OLD/'sample_level_mechanism.csv').set_index('sample_id')
    e=pd.read_csv(OLD/'coverage_transition.csv')
    assert len(m)==m.sample_id.nunique()==len(s)==998
    assert set(m.sample_id)==set(s.index)
    assert sha(CP)==CK and sha(PP)==PSHA
    original=torch.load(PP,map_location='cpu',weights_only=True)
    ori={sid:i for i,sid in enumerate(original['sample_ids'])}
    assert set(ori)==set(s.index)
    for row in m.to_dict('records'):
        path=ROOT/row['path']
        if not path.is_file():
            jnew(OUT/'blocked.json',dict(status='BLOCKED_CACHE_MISSING',missing=str(path.relative_to(ROOT))))
            raise FileNotFoundError(path)
        assert sha(path)==row['sha256'],row['sample_id']
        p=torch.load(path,map_location='cpu',weights_only=True);t=p['tensors'];sid=row['sample_id'];old=s.loc[sid]
        assert p['sample_id']==sid and p['checkpoint_SHA']==CK and p['resize']==.25
        for a in t.values():
            assert a.device.type=='cpu' and torch.isfinite(a).all()
        assert t['indices'].shape==(1,10000) and t['indices'].dtype==torch.int64
        H,W=t['sigmoid'].shape[-2:];idx=t['indices'].numpy().ravel()
        xy=np.stack([idx%W,idx//W],axis=-1).astype(np.float32)
        assert np.array_equal(xy,t['proposals_network'][0,0].numpy())
        assert np.array_equal(xy/.25,t['proposals_raw'][0,0].numpy())
        assert np.array_equal(t['centers_network'].numpy()/.25,t['centers_raw'].numpy())
        assert t['centers_raw'].shape==(19,2)
        assert torch.equal(t['goals'][0,0],t['softargmax_raw'])
        assert torch.equal(t['goals'][1:,0],t['centers_raw'])
        assert torch.equal(t['goals'],original['goals'][:,ori[sid]:ori[sid]+1])
        assert torch.equal(t['paths'],original['paths'][:,ori[sid]:ori[sid]+1])
        for name,key in [('sigmoid','probability_SHA'),('normalized_weights','weights_SHA'),
                         ('proposals_network','proposal_SHA'),('indices','indices_SHA'),
                         ('centers_network','centers_SHA'),('goals','goal_SHA'),('paths','path_SHA')]:
            assert tensor_sha(t[name])==row[key],(sid,key)
        gt=np.array([old.GT_x,old.GT_y]);P=t['proposals_raw'][0,0].numpy().astype(float)
        G=t['goals'][:,0].numpy().astype(float)
        assert abs(np.linalg.norm(P-gt,axis=-1).min()-old.ProposalMinDist)<1e-8
        assert abs(np.linalg.norm(G-gt,axis=-1).min()-old.FinalMinDist20)<1e-8
    counts=s.category25.value_counts().to_dict()
    assert counts=={'both_hit':834,'proposal_hit_final_miss':86,'zero_support_final_miss':77,'positive_support_sampling_miss_final_miss':1}
    for r,n in [(10,489),(25,164),(50,99)]:
        q=e[e.radius==r].set_index('sample_id');assert set(q.index)==set(s.index) and int((~q.C).sum())==n
    identities={k:sorted(s.index[s.category25==k].tolist()) for k in counts}
    source_files=[OLD/n for n in ['tensor_manifest.csv','sample_level_mechanism.csv','coverage_transition.csv','proposal_coverage.csv',
                 'probability_mass.csv','softargmax_kmeans_comparison.csv','threshold_support_details.csv']]
    cfg=dict(stage='Stage11B-2D',scope='EXPLORATORY_DEVELOPMENT_ANALYSIS',CPU_ONLY=True,base_commit=BASE,
      checkpoint_SHA=CK,original_prediction_SHA=PSHA,official_YNet_commit='a345ada1557f53dc2acb4c07f71d340d366980da',
      source_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in source_files],
      original_kmeans_source=dict(path=str(KM.relative_to(ROOT)),sha256=sha(KM)),
      recordings=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'],identities=identities,
      radii=[10,25,50],primary_radius=25,unit='original SDD reference pixels',resize=.25,
      assignment='FINAL_CENTER_VORONOI_ASSIGNMENT',assignment_math='argmin of float64 squared Euclidean distance to the 19 frozen centers; ties lowest head index',
      chunk_size=512,proposal_count=10000,center_count=19,goal_count=20,
      local_count='all draws within <= radius, including repeated draws; unique positions separately',
      cluster_population='all draws with final-center nearest assignment; not original last-iteration labels',
      radius_stats=['mean','median','P90','max'],dispersion='population RMS distance from empirical mean; pairwise center mean/min/median/max',
      local_assigned_population='union of touched clusters and local-draw-weighted cluster population reported separately',
      descriptive_indicators={'local_count_le10':10,'local_count_le100':100,'dominant_local_fraction':.8,
        'large_cluster':'population >= per-target median over nonempty clusters; no universal density claim',
        'local_minority_in_cluster':'local25_count / cluster_population <= .1'},
      comparisons=['failure86_vs_success834','all998_window_weighted','five_recording_equal_weight','deathCircle443','other4_555','all_five_individual_recordings'],
      correlations='Spearman rank correlation without p-values, all998 and each posthoc outcome group; descriptive only',
      quantiles=[0,.1,.25,.5,.75,.9,1],motion_groups=['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group'],
      raw_zero_support_review='all77 unchanged Sigmoid; raw mass, max ratio and removed positive pixels; no resampling',
      analytic_counterfactual='ANALYTIC_COUNTERFACTUAL_ONLY: 1-(1-unthresholded_normalized_mass)^10000',
      case_rule='reuse Stage2C compression-failure frozen q25/q50/q75 sample IDs; plot all three, no visual replacement',
      case_ids=pd.read_csv(OLD/'mechanism_case_manifest.csv').query("category == 'proposal_hit_final_miss'").sample_id.tolist(),
      literature_scope={'required':['Y-Net','PECNet','DLow','AgentFormer','TUTR'],
        'years_recent':[2024,2025,2026],'queries':['goal candidate compression probability coverage trajectory','KMeans trajectory postprocessing',
        'probability weighted representative selection k-center k-medoids farthest point DPP','fixed K trajectory coverage quantization rate distortion'],
        'sources':'author full texts and official code first; arXiv/CrossRef; MCP absent fallback recorded; no abstract-to-fulltext upgrading'},
      figure_contract={'conclusion':'describe how frozen local draws are represented, without causal or performance claims',
        'backend':'python','archetype':'quantitative grid plus spatial case panels','formats':['PNG','SVG'],
        'source':'actual cache coordinates; GT offline only','statistics':'n and ECDF/quantiles; no independent-sample significance'},
      decisions={'observed':'direct cache identity or measured geometry','inferred':'association only',
        'unverified':'new algorithm effects, novelty, semantic causes','not_supported':'KMeans bug or guaranteed ADE gains'},
      future_gate='At most recommend one registered ordinary-method control; no novel-algorithm GO without distinct math and independent confirmation plan',
      restrictions=['no model forward','no optimizer','no CUDA','no new proposals or goals','no threshold change','no test data'])
    jnew(CONF/'analysis_protocol.json',cfg)
    scripts=sorted((ROOT/'scripts/stage11b2d').glob('*.py'))
    jnew(OUT/'protocol_frozen.json',dict(protocol_SHA=sha(CONF/'analysis_protocol.json'),base=BASE,
      numerical_source_SHA=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in scripts],
      protected_list_SHA=sha(CONF/'protected_files.csv.gz'),freeze_code_commit=commit()))
    jnew(OUT/'cache_audit.json',dict(passed=True,targets=998,identities=counts,all_tensor_files_SHA=True,
      all_proposal_order_and_coordinate_mapping=True,original20goals_paths_bitwise=True,
      checkpoint_SHA=sha(CP),original_prediction_SHA=sha(PP),original_25px_identities=True,
      CUDA_initialized=torch.cuda.is_initialized(),GPU_inference=0,training_steps=0,wall_seconds=time.perf_counter()-start))
    print('CACHE ADMISSION PASS 998; protocol frozen',flush=True)

if __name__=='__main__':main()
