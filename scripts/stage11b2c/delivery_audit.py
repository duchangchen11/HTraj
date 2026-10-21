"""Final CPU-only artifact audit; no prediction, training, or test-set score."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import torch,numpy as np,pandas as pd,xml.etree.ElementTree as ET,shutil

def main():
    assert not torch.cuda.is_initialized();frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in frozen['executable_files'])
    assert sha(CONF/'analysis_protocol.json')==frozen['analysis_protocol_SHA']
    assert sha(ROOT/'results/tables/stage11b2a/motion_features.csv')==frozen['historical_motion_labels_SHA']
    assert sha(CP)==CK and sha(PP)==PSHA and json.loads((OUT/'history_after.json').read_text())['passed']
    root=ET.parse(OUT/'full_tests_final.xml');suites=list(root.iter('testsuite'));assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
    skip=[dict(test=c.get('name'),reason=c.find('skipped').get('message')) for c in root.iter('testcase') if c.find('skipped') is not None]
    manifest=pd.read_csv(OUT/'tensor_manifest.csv');assert len(manifest)==998
    old=torch.load(PP,map_location='cpu',weights_only=True);indices={s:i for i,s in enumerate(old['sample_ids'])}
    for row in manifest.to_dict('records'):
        p=ROOT/row['path'];assert sha(p)==row['sha256'];a=torch.load(p,map_location='cpu',weights_only=True);x=a['tensors'];i=indices[row['sample_id']]
        assert torch.equal(x['goals'],old['goals'][:,i:i+1]) and torch.equal(x['paths'],old['paths'][:,i:i+1])
        assert x['indices'].dtype==torch.int64 and x['indices'].numel()==10000 and x['centers_raw'].shape==(19,2)
        for k,t in x.items():
            assert torch.isfinite(t).all(),(row['sample_id'],k)
            if k!='indices':assert t.dtype==torch.float32,(k,t.dtype)
        assert set(a)=={'sample_id','scene','checkpoint_SHA','trace','tensors','reference','resize','unit'}
    figs=json.loads((OUT/'figure_manifest.json').read_text())
    for r in figs:
        p=FIG/(r['figure']+'.svg');ET.parse(p);assert sha(p)==r['SVG_SHA'] and sha(FIG/(r['figure']+'.png'))==r['PNG_SHA']
    fields=['scripts/stage11b2c/','configs/stage11b2c/','reports/stage11b2c/','results/tables/stage11b2c/','results/figures/stage11b2c/']
    changed=subprocess.check_output(['git','diff',BASE,'--name-only'],text=True,cwd=ROOT).splitlines();assert all(any(n.startswith(p) for p in fields) for n in changed)
    files=[p for base in fields for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.suffix.lower() not in {'.pt','.pth','.pkl','.ckpt','.key','.pem'} and p.name!='.env' and p.stat().st_size<=10*1024*1024 for p in files)
    b=json.loads((OUT/'gpu_budget.json').read_text());assert b['new_GPU_seconds']==parent_seconds()<7200
    d=json.loads((OUT/'final_decision.json').read_text());assert d['final_A25_count']==164 and d['proposal_covered']==920 and d['final_covered']==834
    jnew(OUT/'delivery_audit.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,all_history_SHA_unchanged=True,
        history_files=json.loads((OUT/'history_after.json').read_text())['files'],all_frozen_numerical_files_unchanged=True,checkpoint_SHA=CK,original_prediction_SHA=PSHA,
        all998_cache_file_SHA_dtype_finite_and_old_goals_paths_verified=True,all998_reverse_bitwise=True,original_sampler_unchanged=True,
        training_steps=0,test_scoring=False,new_GPU_seconds=parent_seconds(),figure_pairs=len(figs),all_SVG_parse=True,
        tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']},skipped_tests=skip,
        full_test_collection_error_preserved='full_tests_failed_collection.txt/xml; resolved only by renaming new colliding test filename',
        no_large_cache_data_or_weights_in_delivery=True,free_disk_bytes=shutil.disk_usage(CACHE).free,source_commit=commit()))
    p=OUT/'delivery_audit.json';files.append(p)
    cnew(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files)) if p.name!='delivery_SHA256.csv'])
    print('DELIVERY PASS',len(files),'artifacts;',len(figs),'figure pairs;GPU',parent_seconds(),flush=True)
if __name__=='__main__':main()
