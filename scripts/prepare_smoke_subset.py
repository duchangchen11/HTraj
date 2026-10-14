import json
from pathlib import Path
import numpy as np
import hashlib

config = json.loads(Path('configs/stage2_smoke.json').read_text())
manifest_path = Path(config['data_manifest'])
manifest = json.loads(manifest_path.read_text())
random = np.random.default_rng(config['seed'])
result = dict(seed=config['seed'], held_out_scene=config['held_out_scene'],
              data_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              test_used=False, purpose='equal-budget smoke only')
for split, limit in [('train', config['train_windows']), ('validation', config['validation_windows'])]:
    indices = sorted(random.choice(len(manifest['splits'][split]), limit, replace=False).tolist())
    result[split] = dict(indices=indices, sample_ids=[manifest['splits'][split][i]['sample_id'] for i in indices])
path = Path(config['subset_manifest'])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(path)
