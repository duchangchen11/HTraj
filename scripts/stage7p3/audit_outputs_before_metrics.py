"""Global saved-output integrity pass BEFORE reading any test errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import time
import numpy as np
from src.agentformer_official_evaluation.runtime import *
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

start = time.perf_counter()
completion = json.loads((OUT/'inference_completion.json').read_text())
assert completion['all_five_succeeded'], 'Preserve incomplete scenes; report partial admission separately'
verify_inputs()
windows = read_csv(ROOT/'configs/stage7p3/test_windows.csv.gz')
cases = []
for fold in FOLDS:
    rows = [r for r in windows if r['fold'] == fold]
    root = CACHE/'predictions'/fold
    raw = {s: np.genfromtxt(DATA/fold/(s+'.txt'), dtype=str)[:,[0,1,13,15]].astype(np.float32) for s in SEQUENCES[fold]}
    expected = {str(Path(r['sequence'])/f"frame_{int(r['last_observed_frame']):06d}") for r in rows}
    actual = {str(p.relative_to(root/'samples')) for s in (root/'samples').iterdir() for p in s.iterdir()}
    assert expected == actual
    counts = dict(windows=0, target_windows=0, candidate_files=0, predicted_points=0)
    for row in rows:
        frame, ids = int(row['last_observed_frame']), sorted(json.loads(row['target_ids']))
        p, y, serialized = aligned_candidates(sorted((root/'samples'/row['sequence']/f'frame_{frame:06d}').iterdir()), ids, frame, raw[row['sequence']])
        # No displacement error or winner is computed here.
        assert p.shape == (len(ids),20,12,2) and y.shape == (len(ids),12,2)
        for kind in ['recon', 'gt']:
            other = root/kind/row['sequence']/f'frame_{frame:06d}.txt'
            a = np.loadtxt(other, ndmin=2)
            assert a.shape == (len(ids)*12,4) and np.isfinite(a).all()
            assert {(int(t),int(i)) for t,i in a[:,:2]} == {(t,i) for i in ids for t in range(frame+1,frame+13)}
        counts['windows'] += 1
        counts['target_windows'] += len(ids)
        counts['candidate_files'] += 20
        counts['predicted_points'] += len(ids)*20*12
    cases.append(dict(fold=fold, **counts, all_ID_frame_candidate_checks_passed=True))
json_new(OUT/'global_output_pre_metric_validation.json',dict(cases=cases, all_five_passed=True,
    no_ADE_FDE_computed=True, no_candidate_reordering_or_selection=True,
    source_SHA256=digest(__file__), elapsed_seconds=time.perf_counter()-start))
print('Global output validation passed; no prediction errors computed:', cases, flush=True)
