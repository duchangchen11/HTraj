"""Final read-only hash and delivery checks; no old artifact writes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
import time
from src.agentformer_official_evaluation.runtime import ROOT, OUT, digest, verify_inputs
from src.socialvae_baseline.artifacts import json_new

start = time.perf_counter()
protected = json.loads((ROOT/'configs/stage7p3/protected_artifacts.json').read_text())['files']
failures = [r['path'] for r in protected if not (ROOT/r['path']).is_file() or digest(ROOT/r['path']) != r['sha256']]
inputs = verify_inputs()
new_files = subprocess.check_output(['git','ls-files','--others','--exclude-standard'], text=True).splitlines()
large = [p for p in new_files if Path(p).stat().st_size > 10*1024*1024]
for p in new_files:
    assert Path(p).suffix.lower() not in {'.p','.pt','.pth','.ckpt','.zip','.npz','.npy'}, p
assert not large, large
json_new(OUT/'preservation_and_delivery.json', dict(protected_files=len(protected), old_SHA_failures=failures,
    current_frozen_inputs=inputs, new_files_over_10MiB=large, elapsed_seconds=time.perf_counter()-start,
    predictions_checkpoints_data_ignored=True, automatic_push=False))
assert not failures, failures
print('Historical hashes preserved:', len(protected), 'files', flush=True)
