"""Inspect and fingerprint new lightweight files, excluding ignored data/weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import gzip,re,subprocess
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

root=Path('results/tables/stage6p1');out=root/'delivery_inventory.json'
paths=[p for folder in ['configs/stage6p1','scripts/stage6p1','reports/stage6p1','results/tables/stage6p1','src/motion_sampling']
       for p in Path(folder).rglob('*') if p.is_file() and p!=out]
paths+=list(Path('experiments/stage6p1').glob('*.txt'))+[Path('experiments/stage6p1/.gitignore'),Path('tests/test_motion_sampling.py')]
# Do not fingerprint this command's redirected stdout while it is still open.
paths=sorted(set(p for p in paths if str(p)!='experiments/stage6p1/inventory_stdout.txt'
                 and subprocess.run(['git','check-ignore','-q',str(p)]).returncode!=0))
markers=re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}')
records=[]
for p in paths:
    assert p.stat().st_size<5_000_000,p
    assert p.suffix not in ['.pt','.pth','.ckpt','.npz','.npy','.pkl','.pickle'],p
    assert not {'cache','raw','checkpoints','venv'}&set(p.parts),p
    content=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    assert not markers.search(content),p
    records.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)))
json_new(out,dict(files=records,files_excluding_inventory=len(records),total_bytes=sum(r['bytes'] for r in records),
    max_file_bytes=max(r['bytes'] for r in records),private_key_and_known_token_marker_matches=0,
    safety_scope='Size/filename/high-confidence credential markers, not universal secret detection',
    raw_trajectories_weights_caches_environments_ignored=True,
    self_output_log_excluded='experiments/stage6p1/inventory_stdout.txt (open during inventory; checked separately at staging)',
    inventory_self_SHA='Containing delivery Git commit'))
print('Delivery artifacts',len(records),'bytes',sum(r['bytes'] for r in records))
