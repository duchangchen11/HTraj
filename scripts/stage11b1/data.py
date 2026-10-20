"""Identity audit and physically separate train/validation artifacts. No test loader."""
from .common import *
from scripts.stage11ap.safe_data import load_data_pickle
from scripts.stage11b0.data_audit import canonical_sha
import numpy as np

def load(split):
    assert split in {'train','validation'}
    a = json.loads((OUT/'data_audit.json').read_text())['artifacts'][split]
    p = ROOT/a['path']; assert sha(p) == a['sha256']
    df,_ = load_data_pickle(p)
    return df

def manifest(split):
    assert split in {'train','validation'}
    return list(csv.DictReader((OUT/f'{split}_target_manifest.csv').open()))

def main():
    source = ROOT/'results/tables/stage11ap/development_target_manifest_candidate.csv'
    assert sha(source) == '71d36755136cd8abf317b7077e3de9899ac03ad0f519c5a1490b0e3b36afbb12'
    rows = list(csv.DictReader(source.open()))
    idx = {r['sample_id']:r for r in rows}; assert len(idx) == 11323
    sets = {s:{r['scene'] for r in rows if r['proposed_split']==s} for s in ['development_train','development_validation','sealed_test']}
    assert all(not sets[a]&sets[b] for a in sets for b in sets if a!=b)
    assert sets['development_validation'] == {'bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'}
    df,_ = load_data_pickle(RES/'data/SDD/train_trajnet.pkl')
    reference = {}; verified = []
    for mid,g in df.groupby('metaId', sort=False):
        assert len(g)==20 and g.sceneId.nunique()==g.trackId.nunique()==1
        scene = str(g.sceneId.iloc[0]); frame = g.frame.to_numpy(dtype=np.int64)
        assert np.array_equal(np.diff(frame), np.full(19,12))
        sid = f'SDD:{scene}:track{int(g.trackId.iloc[0])}:frame{frame[0]}'
        r = idx[sid]; xy = g[['x','y']].to_numpy(dtype=float)
        assert canonical_sha(xy[:8])==r['history_SHA'] and canonical_sha(xy[8:])==r['GT_SHA']
        assert int(mid)==int(r['meta_id'])
        if scene not in reference:
            reference[scene] = sha(RES/'data/SDD/train'/scene/'reference.jpg')
        assert reference[scene]==r['reference_SHA']
        verified.append(sid)
    assert len(verified)==8494 and len(set(verified))==8494
    artifacts = {}
    for split,tag,count,nscene in [('train','development_train',7496,25),('validation','development_validation',998,5)]:
        rr = [r for r in rows if r['proposed_split']==tag]
        assert len(rr)==count
        cnew(OUT/f'{split}_target_manifest.csv', rr)
        sub = df[df.sceneId.isin(sets[tag])].copy()
        assert len(sub)==20*count and sub.sceneId.nunique()==nscene
        path = CACHE/f'{split}_only.pkl'; assert not path.exists(); sub.to_pickle(path)
        artifacts[split] = dict(path=str(path.relative_to(ROOT)),sha256=sha(path),targets=count,recordings=nscene,
                               manifest_SHA=sha(OUT/f'{split}_target_manifest.csv'))
    jnew(OUT/'data_audit.json', dict(passed=True,protocol='HTraj-YNet-SDD-Development-v1',
         status='PROTOCOL_DEVELOPMENT_ONLY',source_manifest_SHA=sha(source),
         author_source_train_SHA=sha(RES/'data/SDD/train_trajnet.pkl'),
         verified_history_and_GT_targets=8494,artifacts=artifacts,reference_SHA=reference,
         sealed_test_targets=2829,sealed_test_recordings=17,test_pickle_loaded=False,
         recording_overlap=False,window_overlap=False,independent_unseen_site_claim=False,
         natural_person_cross_recording_identity_unverified=True,
         limitations=['SDD numeric semantic ID/name mapping unverified','Exact segmentation checkpoint training membership unverified',
                     'Original short-pool generation manifest partial','Carved validation may have semantic pretraining exposure']))
    print('Data identity PASS: train7496/25 validation998/5; sealed test NOT loaded',flush=True)

if __name__ == '__main__': main()
