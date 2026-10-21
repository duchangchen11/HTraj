"""Identity and frozen-metric reproduction only; no subgroup analysis here."""
from .common import *
import numpy as np,torch,pandas as pd,traceback,cv2
from scripts.stage11b1c.data import load,manifest
from scripts.stage11b0.data_audit import canonical_sha
from scripts.stage11b0.common import RES
def main():
    cpu_guard();result=dict(passed=False,status='BLOCKED',test_loaded=False,CPU_only=True)
    try:
        for p in [PP,CP]:
            if not p.exists():raise FileNotFoundError('Frozen artifact missing: '+str(p.relative_to(ROOT)))
        assert sha(PP)==PSHA and sha(CP)==CK,'Frozen prediction/checkpoint SHA mismatch'
        names=['final_decision.json','paired_epoch030.csv','best_paired_baseline_comparison.csv','baseline_scene_comparison.csv',
            'prediction_geometry_audit.csv','validation_metrics.csv','qualitative_cases.json','validation_target_manifest.csv']
        delivery={r['path']:r['sha256'] for r in csv.DictReader((OLD/'delivery_SHA256.csv').open())};sources={}
        for name in names:
            p=OLD/name;relative=str(p.relative_to(ROOT));assert sha(p)==delivery[relative],name;sources[relative]=sha(p)
        p=torch.load(PP,map_location='cpu',weights_only=True);assert p['paths'].shape==(20,998,12,2) and p['goals'].shape==(20,998,2)
        ids=p['sample_ids'];assert len(ids)==len(set(ids))==998 and p['checkpoint_SHA']==CK and p['units']=='original reference pixels'
        paths=p['paths'].numpy().astype(float);goals=p['goals'].numpy().astype(float);assert np.isfinite(paths).all() and np.isfinite(goals).all()
        df=load('validation');rr=manifest('validation');index={r['sample_id']:r for r in rr};assert set(ids)==set(index)
        groups={int(k):g for k,g in df.groupby('metaId',sort=False)};truth=[];references={}
        for s in ids:
            r=index[s];g=groups[int(r['meta_id'])];xy=g[['x','y']].to_numpy(dtype=float)
            assert len(xy)==20 and canonical_sha(xy[:8])==r['history_SHA'] and canonical_sha(xy[8:])==r['GT_SHA']
            assert r['proposed_split']=='development_validation' and np.all(np.diff(g.frame)==12)
            scene=r['scene']
            if scene not in references:
                q=RES/'data/SDD/train'/scene/'reference.jpg';assert sha(q)==r['reference_SHA'];im=cv2.imread(str(q));assert im is not None
                references[scene]=dict(sha256=sha(q),height=int(im.shape[0]),width=int(im.shape[1]),diagonal=float(np.hypot(*im.shape[:2])))
            truth.append(xy[8:])
        y=np.stack(truth);d=np.linalg.norm(paths-y[None],axis=-1);ge=np.linalg.norm(goals-y[None,:,-1],axis=-1)
        calculated=dict(ADE1=d[0].mean(-1),FDE1=d[0,:,-1],minADE20=d.mean(-1).min(0),minFDE20=d[:,:,-1].min(0),GoalFDE1=ge[0],minGoalFDE20=ge.min(0))
        old=pd.read_csv(OLD/'paired_epoch030.csv').set_index('sample_id').loc[ids];errors={}
        for k,a in calculated.items():
            errors[k]=float(np.max(np.abs(a-old[k].to_numpy())));assert errors[k]<=1e-8
        assert np.array_equal(d.mean(-1).argmin(0),old.best_ADE_head) and np.array_equal(d[:,:,-1].argmin(0),old.best_FDE_head)
        result.update(passed=True,status='PASS',predictions=998,K=20,future_steps=12,goals=20,checkpoint_SHA=CK,prediction_SHA=PSHA,
            original_unit='original reference pixels',all_history_GT_SHA_match=True,max_metric_differences=errors,tolerance_pixels=1e-8,
            original_source_split_column='train denotes author source pool; proposed_split=development_validation controls HTraj membership',
            references=references,sources=sources,source_data_artifact=readj(OLD/'data_audit.json')['artifacts']['validation'],CUDA_initialized=torch.cuda.is_initialized())
    except Exception as e:
        result.update(error_type=type(e).__name__,error=str(e));traceback.print_exc();raise
    finally:writej(OUT/'data_cache_audit.json',result)
    print('Frozen cache/data/paired metrics PASS;998 targets,20 paths/goals; no test or model execution',flush=True)
if __name__=='__main__':main()
