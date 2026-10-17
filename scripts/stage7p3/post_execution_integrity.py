"""Read-only final integrity, source/version and executed-command receipt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
import time
from src.agentformer_official_evaluation.runtime import *
from src.socialvae_baseline.artifacts import json_new

start = time.perf_counter()
protected = json.loads((ROOT/'configs/stage7p3/protected_artifacts.json').read_text())['files']
failures = [r['path'] for r in protected if not (ROOT/r['path']).is_file() or digest(ROOT/r['path']) != r['sha256']]
inputs = verify_inputs()
for fold in FOLDS:
    assert subprocess.run(['git','check-ignore','-q',str(CACHE/'predictions'/fold)],cwd=ROOT).returncode == 0
sources = [*Path('scripts/stage7p3').glob('*.py'), *Path('src/agentformer_official_evaluation').glob('*.py'), Path('tests/test_agentformer_official_evaluation.py')]
json_new(OUT/'post_execution_integrity.json', dict(historical_files_checked=len(protected), historical_SHA_failures=failures,
    frozen_inputs=inputs, seconds=time.perf_counter()-start,
    delivery_source_SHA256={str(p): digest(p) for p in sources}, no_prior_stage_files_modified=True))
assert not failures, failures
print('Final historical SHA check:',len(protected),'unchanged files; frozen code/data/weights unchanged',flush=True)
