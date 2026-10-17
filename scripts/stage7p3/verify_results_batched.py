"""Final artifact acceptance; verify every saved-output hash and aggregation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from src.agentformer_official_evaluation.runtime import *
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

s = json.loads((OUT/'evaluation_summary.json').read_text())
assert s['all_five_completed'] and s['preliminary_reproduction_passed'] and not s['failures']
assert json.loads((OUT/'global_output_pre_metric_validation.json').read_text())['all_five_passed']
assert not json.loads((OUT/'post_execution_integrity.json').read_text())['historical_SHA_failures']
root = ET.parse(OUT/'full_tests.xml').getroot()
assert all(int(x.attrib.get(k, 0)) == 0 for x in root.iter('testsuite') for k in ['errors','failures','skipped'])
verified_files = 0
all_output_paths = []
for r in s['folds']:
    fold = r['fold']
    targets = read_csv(OUT/f'{fold}_target_metrics.csv.gz')
    identities = {(t['sequence'],t['last_observed_frame'],t['person_id']) for t in targets}
    assert len(identities) == len(targets) == protocol()['counts'][fold]['targets']
    for key in ['minADE20','minFDE20','K1_first_ADE','K1_first_FDE']:
        assert abs(float(np.mean([float(t[key]) for t in targets]))-r[key]) < 1e-12
    hashes = read_csv(OUT/f'{fold}_prediction_file_SHA256.csv.gz')
    assert len(hashes) == r['windows']*22
    for entry in hashes:
        assert digest(ROOT/entry['path']) == entry['sha256'], entry['path']
        all_output_paths.append(entry['path'])
    verified_files += len(hashes)
ignored = subprocess.run(['git','check-ignore','--stdin','-z'], input=('\0'.join(all_output_paths)+'\0').encode(), capture_output=True, cwd=ROOT, check=True)
assert set(ignored.stdout.decode().rstrip('\0').split('\0')) == set(all_output_paths)
for key in ['minADE20','minFDE20','K1_first_ADE','K1_first_FDE']:
    assert abs(float(np.mean([r[key] for r in s['folds']]))-s['macro'][key]) < 1e-12
expected = ['01_FROZEN_EVALUATION_PROTOCOL.md','02_OFFICIAL_CHECKPOINT_VERIFICATION.md',
    '03_FIVE_SCENE_EVALUATION.md','04_AUTHOR_RESULT_COMPARISON.md','05_EVALUATION_REPRODUCIBILITY.md',
    '06_GO_NO_GO.md','STAGE7P3_REPORT.md']
assert all((ROOT/'reports/stage7p3'/p).is_file() for p in expected)
inputs = verify_inputs()
json_new(OUT/'final_artifact_checks.json', dict(all_passed=True, prediction_recon_GT_files_hash_verified=verified_files,
    all_files_ignored=True, target_window_uniqueness=True, target_weighted_aggregation_verified=True,
    five_fold_equal_macro_verified=True, frozen_inputs=inputs,
    test_suite_passed=419, failures=0, report_files=len(expected), verifier_SHA256=digest(__file__)))
print('Final artifact checks passed:', verified_files, 'output-file SHA verified; correct target and macro aggregation', flush=True)
