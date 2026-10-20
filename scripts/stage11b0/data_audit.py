"""Recheck frozen identities. Test pickle is hashed as opaque bytes, never loaded."""
from .common import *
from scripts.stage11ap.safe_data import load_data_pickle
import numpy as np,cv2
def canonical_sha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def load_train():
    artifact=OUT/'training_data_artifact.json'
    if artifact.exists():
        record=json.loads(artifact.read_text());path=ROOT/record['path']
        assert sha(path)==record['sha256']
        df,_=load_data_pickle(path)
        assert len(df)==7496*20 and df.sceneId.nunique()==25
        return df
    cfg=json.loads((ROOT/'configs/stage11ap/protocol_candidate.json').read_text())
    df,_=load_data_pickle(RES/'data/SDD/train_trajnet.pkl')
    scenes=set(cfg['proposed_protocol']['train_videos'])
    return df[df.sceneId.isin(scenes)].copy()
def main():
    previous=ROOT/'results/tables/stage11ap/development_target_manifest_candidate.csv'
    cfg=json.loads((ROOT/'configs/stage11ap/protocol_candidate.json').read_text())
    assert sha(previous)==cfg['identity']['manifest_SHA']
    rows=list(csv.DictReader(previous.open()))
    ids=[r['sample_id'] for r in rows];assert len(ids)==len(set(ids))
    split_scenes={k:{r['scene'] for r in rows if r['proposed_split']==k} for k in ['development_train','development_validation','sealed_test']}
    counts={k:sum(r['proposed_split']==k for r in rows) for k in split_scenes}
    assert counts==dict(development_train=7496,development_validation=998,sealed_test=2829)
    for a in split_scenes:
        for b in split_scenes:
            if a!=b:assert not split_scenes[a]&split_scenes[b]
    index={r['sample_id']:r for r in rows}
    df,_=load_data_pickle(RES/'data/SDD/train_trajnet.pkl')
    verified=[];outside=[];images={}
    for mid,g in df.groupby('metaId',sort=False):
        assert len(g)==20 and g.sceneId.nunique()==g.trackId.nunique()==1
        scene=g.sceneId.iloc[0];track=int(g.trackId.iloc[0]);frames=g.frame.to_numpy(dtype=np.int64)
        assert np.array_equal(np.diff(frames),np.full(19,12))
        sid=f'SDD:{scene}:track{track}:frame{frames[0]}';r=index[sid]
        assert r['scene']==scene and int(r['last_frame'])==int(frames[-1])
        xy=g[['x','y']].to_numpy(dtype=float)
        assert canonical_sha(xy[:8])==r['history_SHA'] and canonical_sha(xy[8:])==r['GT_SHA']
        if scene not in images:
            image=RES/'data/SDD/train'/scene/'reference.jpg'
            assert sha(image)==r['reference_SHA']
            im=cv2.imread(str(image));small=cv2.resize(im,(0,0),fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
            images[scene]=(im.shape[:2],tuple(int(np.ceil(n/32)*32) for n in small.shape[:2]))
        (H,W),(Hp,Wp)=images[scene]
        for t,(x,y) in enumerate(xy):
            if x<0 or y<0 or x>=W or y>=H:
                q=np.round(np.array([x,y],dtype='float32')*.25).astype(int)
                outside.append(dict(sample_id=sid,split=r['proposed_split'],scene=scene,frame=int(frames[t]),x=x,y=y,
                                    width=W,height=H,rounded_scaled_x=int(q[0]),rounded_scaled_y=int(q[1]),
                                    padded_width=Wp,padded_height=Hp,within_padded_grid=bool(0<=q[0]<Wp and 0<=q[1]<Hp),
                                    clipped=False,kept=True))
        verified.append(r)
    assert len(verified)==8494 and len(outside)==2
    assert all(r['within_padded_grid'] and r['kept'] and not r['clipped'] for r in outside)
    expected=list(csv.DictReader((ROOT/'results/tables/stage11ap/extracted_resource_SHA256.csv').open()))
    test_row=next(r for r in expected if r['path'].endswith('data/SDD/test_trajnet.pkl'))
    test_path=ROOT/'experiments/stage11ap/cache/author_resources'/test_row['path']
    assert sha(test_path)==test_row['sha256']
    trainrows=[r for r in rows if r['proposed_split']=='development_train']
    cnew(OUT/'train_target_manifest.csv',trainrows)
    cnew(OUT/'out_of_image_points.csv',outside)
    stats=[]
    for scene in sorted(split_scenes['development_train']):
        rr=[r for r in trainrows if r['scene']==scene];(h,w),(hp,wp)=images[scene]
        stats.append(dict(scene=scene,targets=len(rr),raw_height=h,raw_width=w,padded_height=hp,padded_width=wp,
                          notebook_target_batch4_steps=int(np.ceil(len(rr)/4)),yaml_target_batch8_steps=int(np.ceil(len(rr)/8))))
    cnew(OUT/'train_scene_inventory.csv',stats)
    jnew(OUT/'split_audit.json',dict(status='SPLIT_IDENTITY_PASS',full_protocol_ready=False,counts=counts,
        recordings={k:sorted(v) for k,v in split_scenes.items()},source_manifest_SHA=sha(previous),
        train_manifest_SHA=sha(OUT/'train_target_manifest.csv'),author_train_pickle_SHA=sha(RES/'data/SDD/train_trajnet.pkl'),
        test_pickle_SHA=test_row['sha256'],test_pickle_unpickled=False,test_predictions=False,
        source_train_and_val_GT_SHA_recomputed=8494,test_identity_and_GT_SHA_bound_to_frozen_manifest=True,
        same_window_cross_split=False,time_support_cross_split_on_recording_identity=False,frame_support='20 samples x step12 -> first..first+228 original frames',
        background_sharing_train_val=True,background_sharing_author_train_test=['coupa','gates','hyang','nexus'],
        raw_video_global_time_and_natural_person_cross_recording_identity_unverified=True,
        raw_source_selection_provenance='Author Drive package verified SHA; original raw annotation-to-short-pool recipe remains PARTIAL',
        out_of_image_points_preserved=2,all_samples_retained=True,semantic_mapping='SEMANTIC_CLASS_MAPPING_UNVERIFIED',
        semantic_pretraining='SEGMENTATION_EXPOSURE_UNVERIFIED'))
    print('Split identity PASS; train7496/25, val998/5, test2829/17; 2 outside points retained; no test pickle decoded')
if __name__=='__main__':main()
