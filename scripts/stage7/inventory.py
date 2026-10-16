"""Fingerprint new lightweight Stage7 artifacts; excluded cache stays local."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import gzip,re,subprocess
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

out=Path('results/tables/stage7/delivery_inventory.json')
folders=['configs/stage7','reports/stage7','results/tables/stage7','scripts/stage7','src/public_admission']
paths=[p for folder in folders for p in Path(folder).rglob('*') if p.is_file() and p!=out]
paths+=list(Path('experiments/stage7').glob('*.txt'))+[Path('experiments/stage7/.gitignore'),Path('tests/test_public_admission.py')]
paths=sorted(set(p for p in paths if subprocess.run(['git','check-ignore','-q',str(p)]).returncode!=0))
pattern=re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}')
rows=[]
for p in paths:
    assert p.stat().st_size<5_000_000,p
    assert p.suffix not in ['.pt','.pth','.npy','.npz','.ckpt','.pickle','.pkl'],p
    assert not {'cache','raw','venv','checkpoints'}&set(p.parts),p
    content=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    assert not pattern.search(content),p
    rows.append(dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size))
json_new(out,dict(files=rows,total_bytes=sum(r['bytes'] for r in rows),max_bytes=max(r['bytes'] for r in rows),
    no_raw_data_weights_credentials_or_environment_staged=True,known_credential_markers_found=0,
    scope='Filename/size/high-confidence credential markers, not universal secret detection',
    inventory_self_SHA='Containing Git delivery commit, no circular SHA'))
print('Delivery inventory:',len(rows),'files',sum(r['bytes'] for r in rows),'bytes')
