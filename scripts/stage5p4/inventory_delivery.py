"""Lightweight delivery hashes; excludes ignored raw packets, weights and caches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
roots=['configs/stage5p4','reports/stage5p4','results/tables/stage5p4','scripts/stage5p4','src/socialvae_capacity']
paths={p for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.xml','.txt','.csv','.gz','.md']}
paths|={Path(p) for p in ['tests/test_socialvae_capacity.py','tests/test_socialvae_capacity_decision.py','tests/test_socialvae_placebo.py','experiments/stage5p4/.gitignore']}
paths|=set(Path('experiments/stage5p4').glob('logs_*.txt'))
output=Path('results/tables/stage5p4/delivery_inventory.json');paths.discard(output)
json_new(output,dict(files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(paths)],
 checkpoints='weights excluded from Git; source/selected/periodic checkpoint paths and SHA in each run/checkpoint_inventory.csv',
 raw_data='excluded; frozen raw-file/manifest SHA in configurations and run/provenance.json'))
print('Delivery inventory:',len(paths),'files')
