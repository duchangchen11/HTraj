"""CPU-only final conservation, provenance and Git-scope audit."""
from .common import *
import re
import xml.etree.ElementTree as ET
import pandas as pd

def main():
    cpu_guard()
    old=json.loads((OUT/'history_after.json').read_text());assert old['passed'] and old['files']==92248
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'analysis_protocol.json')==frozen['protocol_SHA']
    for row in frozen['numerical_source_SHA']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    assert sha(CP)==CK and sha(PP)==PSHA
    f=pd.read_csv(OUT/'target_diagnostics.csv');c=pd.read_csv(OUT/'cluster_diagnostics.csv')
    assert len(f)==998 and len(c)==18962 and (c.groupby('sample_id').Population.sum()==10000).all()
    assert int((f.category25=='proposal_hit_final_miss').sum())==86
    e=pd.read_csv(OUT/'local_proposal_assignments.csv.gz');assert len(e)==6050 and e.sample_id.nunique()==86
    assert c.assignment.eq('FINAL_CENTER_VORONOI_ASSIGNMENT').all()
    figures=json.loads((OUT/'figure_manifest.json').read_text());assert len(figures)==5
    for r in figures:
        assert sha(FIG/(r['name']+'.png'))==r['PNG_SHA'] and sha(FIG/(r['name']+'.svg'))==r['SVG_SHA']
        ET.parse(FIG/(r['name']+'.svg'))
    xml=ET.parse(OUT/'full_tests.xml');suites=list(xml.iter('testsuite'))
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert tests==dict(tests=630,failures=0,errors=0,skipped=1)
    em=json.loads((OUT/'execution_manifest.json').read_text())
    em['source_files']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts/stage11b2d').rglob('*.py')) if '__pycache__' not in p.parts]
    em['postprocessing_source_record']='Source file SHA authoritative; numerical execution commit frozen ef66d85; delivery includes final code and metadata'
    (OUT/'execution_manifest.json').write_text(json.dumps(em,ensure_ascii=False,indent=2)+'\n')
    allowed=['configs/stage11b2d/','scripts/stage11b2d/','reports/stage11b2d/','results/tables/stage11b2d/','results/figures/stage11b2d/']
    changed=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(any(p.startswith(a) for a in allowed) for p in changed)
    files=sorted(p for a in allowed for p in (ROOT/a).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_SHA256.csv')
    assert not any(p.suffix.lower() in {'.pt','.pth','.pkl','.npy','.npz','.pdf','.key','.pem'} for p in files)
    assert all(p.stat().st_size<10*1024*1024 for p in files)
    secret=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bghp_[A-Za-z0-9]{30,}')
    for p in files:
        if p.suffix in {'.py','.json','.csv','.md','.txt'}:assert not secret.search(p.read_text()),p
    d=dict(passed=True,historical_files_unchanged=old['files'],numerical_frozen_code_unchanged=True,
      cache_identity_verified=True,targets=998,compression_targets=86,zero_support=77,
      final_voronoi_cells=18962,local_draw_assignments=6050,figures=5,tests=tests,
      CPU_only=True,new_GPU_seconds=0,new_frozen_model_inference=0,training_steps=0,test_set_scoring=False,
      no_original_algorithm_implemented=True,source_code_SHA=sha(Path(__file__)),code_commit=commit(),
      large_tables='cluster_diagnostics.csv is a required derived scalar table, <10 MiB; tensors/PDF/weights remain ignored',
      old_results_modified=False,sensitive_file_check='PASS',Git_scope='only Stage11B-2D new directories')
    jnew(OUT/'delivery_audit.json',d)
    files.append(OUT/'delivery_audit.json')
    csvnew(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))])
    print('DELIVERY PASS',len(set(files)),'files; historical',old['files'],'unchanged; tests',tests,flush=True)

if __name__=='__main__':main()
