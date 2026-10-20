"""Author archive inventory, semantic geometry, identities and split-only audit.

Test coordinates are read ONLY for byte/identity/time-support validation, never
for predictions, errors, ranking or model selection.
"""
from common import *
from safe_data import load_data_pickle
import numpy as np
from PIL import Image
import yaml

RES=CACHE/'author_resources/ynet_additional_files'
def save(path,value):
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(value)), 'Existing evidence differs: '+str(path)
    else:jnew(path,value)
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()
def audit_table(path,split):
    df,globals_=load_data_pickle(path)
    assert set(df.columns)=={'frame','trackId','x','y','sceneId','metaId'}
    assert np.isfinite(df[['frame','trackId','x','y','metaId']].to_numpy()).all()
    records=[];identities={};outside=0;offsets=[]
    for mid,g in df.groupby('metaId',sort=False):
        assert len(g)==20 and g.sceneId.nunique()==1 and g.trackId.nunique()==1
        f=g.frame.to_numpy(dtype=np.int64);xy=g[['x','y']].to_numpy(dtype=float)
        assert np.array_equal(np.diff(f),np.full(19,12)), (mid,f.tolist())
        scene=g.sceneId.iloc[0];track=int(g.trackId.iloc[0]);first=int(f[0])
        image=RES/'data/SDD'/split/scene/'reference.jpg'
        with Image.open(image) as im:w,h=im.size
        o=int(((xy[:,0]<0)|(xy[:,0]>=w)|(xy[:,1]<0)|(xy[:,1]>=h)).sum());outside+=o
        sid=f'SDD:{scene}:track{track}:frame{first}'
        r=dict(sample_id=sid,split=split,scene=scene,site=scene.rsplit('_',1)[0],video=scene.rsplit('_',1)[1],person_id=track,meta_id=int(mid),frame_id=first,last_frame=int(f[-1]),
               frame_step=12,history_SHA=array_sha(xy[:8]),GT_SHA=array_sha(xy[8:]),reference_SHA=sha(image),out_of_image_points=o)
        assert sid not in identities
        identities[sid]=r;records.append(r)
    for (_,track),g in df.groupby(['sceneId','trackId'],sort=True):
        starts=g.groupby('metaId').frame.min().sort_values().to_numpy()
        if len(starts)>1:offsets.extend(np.diff(starts).astype(int).tolist())
    return df,records,dict(rows=len(df),targets=len(records),scenes=sorted(df.sceneId.unique()),globals_allowed=globals_,
                           outside_reference_points=outside,consecutive_window_start_steps=dict(__import__('collections').Counter(offsets)),frame_support_check_passed=True)

def main():
    inventory=[];mask_rows=[];maps=RES/'data/SDD_semantic_maps'
    for p in sorted(RES.rglob('*')):
        if not p.is_file():continue
        rel=str(p.relative_to(RES));entry=dict(path=rel,bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix.lower() in {'.jpg','.png'}:
            with Image.open(p) as im:
                entry.update(width=im.width,height=im.height,mode=im.mode)
                if 'SDD_semantic_maps' in p.parts:
                    a=np.array(im);values,counts=np.unique(a,return_counts=True)
                    scene=p.stem.removesuffix('_mask');split=p.parent.name.removesuffix('_masks')
                    ref=RES/'data/SDD'/split/scene/'reference.jpg'
                    palette=im.getpalette()
                    mr=dict(scene=scene,split=split,width=im.width,height=im.height,mode=im.mode,unique_values=values.tolist(),pixel_counts=counts.tolist(),
                            palette=palette,semantic_SHA=entry['sha256'],reference_exists=ref.exists(),class_name_mapping='UNVERIFIED')
                    if ref.exists():
                        with Image.open(ref) as rim:mr.update(reference_dimensions=list(rim.size),dimensions_match=rim.size==im.size,reference_SHA=sha(ref))
                    mask_rows.append(mr)
        inventory.append(entry)
    cfg=yaml.safe_load((RES/'config/sdd_trajnet.yaml').read_text())
    save(OUT/'official_sdd_config.json',dict(path='config/sdd_trajnet.yaml',sha256=sha(RES/'config/sdd_trajnet.yaml'),values=cfg,obs_len_from_notebook=8,pred_len_from_notebook=12))
    save(OUT/'semantic_map_audit.json',dict(maps=mask_rows,total_maps=len(mask_rows),id_name_mapping_verified=False,
                class_count_config=cfg['semantic_classes'],manual_origin='Author-provided maps; paper describes hand annotation, no per-file creation metadata',
                sixth_class_identity='UNVERIFIED; do not assume background',semantic_pretraining_split_exposure='UNVERIFIED'))
    train,tr,ta=audit_table(RES/'data/SDD/train_trajnet.pkl','train')
    test,te,tb=audit_table(RES/'data/SDD/test_trajnet.pkl','test')
    allrows=tr+te
    if not (OUT/'author_target_manifest.csv').exists():cnew(OUT/'author_target_manifest.csv',allrows)
    train_scenes=set(ta['scenes']);test_scenes=set(tb['scenes'])
    exact=set(r['sample_id'] for r in tr)&set(r['sample_id'] for r in te)
    sites_train={r['site'] for r in tr};sites_test={r['site'] for r in te}
    # A single deterministic candidate before any errors are calculated: select
    # one training recording per physical site by SHA256(scene ID), reserve test.
    val_scenes=[]
    for site in sorted(sites_train):
        videos=sorted(s for s in train_scenes if s.rsplit('_',1)[0]==site)
        if len(videos)<2:continue  # single-recording sites stay in train; do not empty a site
        val_scenes.append(min(videos,key=lambda s:hashlib.sha256(('HTraj-SDD-dev-v1:'+s).encode()).hexdigest()))
    dev=[]
    for r in allrows:
        q=dict(r);q['proposed_split']='sealed_test' if r['split']=='test' else ('development_validation' if r['scene'] in val_scenes else 'development_train')
        dev.append(q)
    cnew(OUT/'development_target_manifest_candidate.csv',dev)
    audit=dict(author_train=ta,author_test=tb,train_test_scene_intersection=sorted(train_scenes&test_scenes),exact_window_intersection=sorted(exact),
               site_background_overlap=sorted(sites_train&sites_test),track_numeric_id_reuse_is_not_same_person=True,
               identity_key='scene + trackId + complete 20 frame support',test_metrics_calculated=False,
               training_validation_definition='Official bundle has only train/test; author notebook uses test for checkpoint selection; forbidden in HTraj',
               candidate_val_scenes=val_scenes,candidate_counts=dict(__import__('collections').Counter(r['proposed_split'] for r in dev)),
               no_time_support_cross_split_overlap=not bool(train_scenes&test_scenes),
               original_raw_provenance='Bound to author Drive archive; original video/complete annotation archive not acquired')
    jnew(OUT/'split_identity_audit.json',audit)
    jnew(OUT/'resource_inventory.json',dict(files=inventory,author_resource_archive='AVAILABLE',
        author_short_SDD_trajectory_pool='AVAILABLE',full_raw_SDD='PARTIAL',author_RGB='AVAILABLE for 47 recordings',
        semantic_masks='AVAILABLE for 60 recordings; palette names UNVERIFIED',segmentation_checkpoint='AVAILABLE bytes; strict loading pending',
        trajectory_checkpoint='AVAILABLE bytes; strict loading pending',validation_split='UNAVAILABLE in original author package',
        original_annotations='NOT included',original_videos='NOT included; 69GB bundle not downloaded',license='SDD CC BY-NC-SA3.0; author archive lacks new license statement; model source MIT'))
    print('Train targets',len(tr),'test identity-only',len(te),'masks',len(mask_rows),'candidate val',val_scenes)
if __name__=='__main__':main()
