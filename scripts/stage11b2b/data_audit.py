"""Verified inherited train/validation pools; no test loader exists."""
from .common import *
from scripts.stage11b1.data import load as previous_load,manifest as previous_manifest
from scripts.stage11b0.data_audit import canonical_sha
import numpy as np,cv2

def manifest(split):return previous_manifest(split)

def load(split):
    assert split in {'train','validation'}
    df=previous_load(split)
    expected=7496 if split=='train' else 998
    assert df.metaId.nunique()==expected and len(df)==expected*20
    return df

def scene_arrays(df,scene):
    import torch
    g=df[df.sceneId==scene];ids=g.metaId.drop_duplicates().astype(int).tolist()
    a=g[['x','y']].to_numpy(dtype='float32').reshape(-1,20,2)
    assert len(ids)==len(a)
    assert (g.metaId.to_numpy().reshape(-1,20)==np.asarray(ids)[:,None]).all()
    return torch.from_numpy(a*.25),ids

def input_records(split):
    """History-only record projection, with no future column in returned API."""
    df=load(split);groups={int(k):g for k,g in df.groupby('metaId',sort=False)};result=[]
    for r in manifest(split):
        h=groups[int(r['meta_id'])][['x','y']].iloc[:8].to_numpy(dtype=float)
        assert canonical_sha(h)==r['history_SHA']
        result.append(dict(sample_id=r['sample_id'],scene=r['scene'],history=h.astype('float32')))
    return result

def truths(split):
    df=load(split);groups={int(k):g for k,g in df.groupby('metaId',sort=False)};result={}
    for r in manifest(split):
        y=groups[int(r['meta_id'])][['x','y']].iloc[8:].to_numpy(dtype=float)
        assert canonical_sha(y)==r['GT_SHA']
        result[r['sample_id']]=y
    return result

def main():
    previous=json.loads((ROOT/'results/tables/stage11b1/data_audit.json').read_text())
    original=ROOT/'results/tables/stage11ap/development_target_manifest_candidate.csv'
    assert sha(original)==previous['source_manifest_SHA']
    scenes={};reference={};outside=[]
    for split,count,nscene in [('train',7496,25),('validation',998,5)]:
        df=load(split);rows=manifest(split);assert len(rows)==count
        groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
        for r in rows:
            g=groups[int(r['meta_id'])];xy=g[['x','y']].to_numpy(dtype=float)
            assert len(g)==20 and g.sceneId.nunique()==g.trackId.nunique()==1
            assert np.array_equal(np.diff(g.frame.to_numpy()),np.full(19,12))
            assert str(g.sceneId.iloc[0])==r['scene'] and int(g.trackId.iloc[0])==int(r['person_id'])
            assert canonical_sha(xy[:8])==r['history_SHA'] and canonical_sha(xy[8:])==r['GT_SHA']
            scene=r['scene']
            if scene not in reference:
                p=RES/'data/SDD/train'/scene/'reference.jpg';im=cv2.imread(str(p));assert im is not None
                reference[scene]=dict(sha256=sha(p),height=im.shape[0],width=im.shape[1])
            info=reference[scene];assert info['sha256']==r['reference_SHA']
            for t,(x,y) in enumerate(xy):
                if not (0<=x<info['width'] and 0<=y<info['height']):
                    outside.append(dict(sample_id=r['sample_id'],split=split,step=t,x=x,y=y,kept=True,clipped=False))
        scenes[split]=sorted(df.sceneId.unique());assert len(scenes[split])==nscene
        cnew(OUT/f'{split}_target_manifest.csv',rows)
    assert not set(scenes['train'])&set(scenes['validation'])
    assert scenes['validation']==['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4']
    assert len(outside)==2 and all(r['split']=='train' for r in outside)
    cnew(OUT/'out_of_image_points.csv',outside)
    # Frozen K1 results are reused only after hashes and every paired ID/GT are checked.
    delivery=list(csv.DictReader((ROOT/'results/tables/stage11b1/delivery_SHA256.csv').open()))
    expected={r['path']:r['sha256'] for r in delivery};baselines={}
    for name in ['baseline_comparison.csv','baseline_paired_errors.csv','baseline_scene_metrics.csv','validation_target_manifest.csv']:
        p=ROOT/'results/tables/stage11b1'/name;assert sha(p)==expected[str(p.relative_to(ROOT))]
        baselines[name]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    index={r['sample_id']:r for r in manifest('validation')}
    pairs=list(csv.DictReader((ROOT/baselines['baseline_paired_errors.csv']['path']).open()))
    assert len(pairs)==998*3
    for r in pairs:
        assert r['sample_id'] in index and r['history_SHA']==index[r['sample_id']]['history_SHA'] and r['GT_SHA']==index[r['sample_id']]['GT_SHA']
    for model in ['CV-Last2','OLS4','OLS8']:
        assert {r['sample_id'] for r in pairs if r['model']==model}==set(index)
    jnew(OUT/'data_audit.json',dict(passed=True,protocol='HTraj-YNet-SDD-Development-v1',status='PROTOCOL_DEVELOPMENT_ONLY',
         source_manifest_SHA=sha(original),artifacts=previous['artifacts'],counts={'train':7496,'validation':998,'sealed_test':2829},
         scenes=scenes,reference=reference,baselines=baselines,all8494_history_GT_SHA_verified=True,
         cross_split_recording_or_window_overlap=False,out_of_image_points_kept=2,error_filtering=False,
         test_loaded=False,test_GPU_entry=False,limits=previous['limitations']))
    print('Data identities and frozen K1 baselines PASS; all targets retained; test sealed',flush=True)

if __name__=='__main__':main()
