"""Verify historical bytes without interpreting any old test predictions/errors."""
import csv
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from stage7p5_prepare import digest


def main():
    start=time.perf_counter()
    protection=json.loads((ROOT/'configs/stage7p5/protected_artifacts.json').read_text())
    protocol=json.loads((ROOT/'configs/stage7p5/protocol.json').read_text())
    failures=[];checked=0
    def check(r):
        nonlocal checked
        p=ROOT/r['path']
        if not p.exists() or digest(p)!=r['sha256']:failures.append(r['path'])
        checked+=1
    for r in protection['files']:check(r)
    test_bytes=0
    for index in protection['prediction_SHA_indices']:
        check(index)
        with gzip.open(ROOT/index['path'],'rt') as f:
            for row in csv.DictReader(f):check(row);test_bytes+=1
    for r in protocol['frozen_inputs']:check(r)
    for index in protocol['prediction_indices']:
        check(index)
        with gzip.open(ROOT/index['path'],'rt') as f:
            for row in csv.DictReader(f):check(row)
    json_new(ROOT/'results/tables/stage7p5/history_preservation.json',dict(
        protected_file_entries=len(protection['files']),old_test_prediction_files_bytes_only=test_bytes,
        total_SHA_checks=checked,failures=failures,passed=not failures,seconds=time.perf_counter()-start,
        old_test_metrics_parsed=False,statement='Old test files hashed as opaque bytes for preservation only; no test errors read or recomputed'))
    if failures:raise RuntimeError('Historical preservation check failed')
    print(json.dumps(dict(checked=checked,passed=True,seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()
