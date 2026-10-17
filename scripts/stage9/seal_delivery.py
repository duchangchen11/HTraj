"""Finalize hashes AFTER verification stdout closes; no experimental data edits."""
import re
from common import *

def run():
    checks=OUT/'delivery_checks.json';data=json.loads(checks.read_text());assert data['passed']
    data.update(final_seal_after_verifier_log_closed=True,final_index_scope='All Stage9 owned deliverables except delivery_SHA256.csv itself',
        source_snapshot_scope='code_and_config_SHA256.csv is pre-evaluation snapshot; final delivery index also covers post-hoc reporting helpers')
    checks.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    files=[]
    for directory in ['configs/stage9','scripts/stage9','reports/stage9','results/tables/stage9']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_SHA256.csv')
    files.append(ROOT/'tests/test_stage9_controlled_pilot.py')
    for p in files:
        assert p.stat().st_size<10_000_000
        assert p.suffix not in {'.pt','.pth','.ckpt','.pkl','.npy','.npz','.pdf','.pem','.key'}
    # Replace only this phase's checksum index, never predictions, model states or metrics.
    temp=OUT/'delivery_SHA256.csv.tmp'
    with temp.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','sha256','bytes'],lineterminator='\n');writer.writeheader()
        for p in sorted(files):writer.writerow(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    temp.replace(OUT/'delivery_SHA256.csv')
    with (OUT/'delivery_SHA256.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in rows)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    print('Final byte SHA sealed after all writer processes closed:',len(rows),'files; raw metrics untouched.')

if __name__=='__main__':run()
