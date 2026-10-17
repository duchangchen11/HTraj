"""Verify Stage8 results and all inherited historical artifacts, then seal delivery."""
import csv
import gzip
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
OUT=ROOT/'results/tables/stage8'
BASE='28a1c50482802ab0ea55a791b22cba56a8e5a6a4'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    protocol=json.loads((ROOT/'configs/stage8/protocol.json').read_text())
    failed=[];count=0
    def check(path,expected):
        nonlocal count
        count+=1
        if not (ROOT/path).is_file() or sha(ROOT/path)!=expected:failed.append(path)
    check(protocol['protection_index'],protocol['protection_index_SHA256'])
    with gzip.open(ROOT/protocol['protection_index'],'rt') as f:
        for r in csv.DictReader(f):check(r['path'],r['sha256'])
    for index in protocol['prediction_SHA_indices']:
        check(index['path'],index['sha256'])
        with gzip.open(ROOT/index['path'],'rt') as f:
            for r in csv.DictReader(f):check(r['path'],r.get('sha256',r.get('SHA256')))
    json_new(OUT/'historical_preservation.json',dict(checked_files=count,failed=failed,passed=not failed,
        test_prediction_bytes_hashed_only=True,test_errors_parsed=False))
    if failed:raise RuntimeError('Historical file changed')
    # Validate all current frozen external sources and author checkpoints too.
    with gzip.open(OUT/'source_and_weight_SHA256.csv.gz','rt') as f:
        for r in csv.DictReader(f):check(r['path'],r['sha256'])
    if failed:raise RuntimeError('New frozen official source/checkpoint changed')
    probe=json.loads((OUT/'resonance_probe.json').read_text())
    author=json.loads((OUT/'resonance_author_checkpoint.json').read_text())
    assert probe['passed'] and author['strict_load'] and author['checkpoint_unchanged']
    assert probe['optimizer_updates']==0 and probe['formal_epochs']==0
    assert probe['pairing_alias']['max_matrix_difference']<1e-5
    assert author['author_weight_pairing_alias_max_difference']<1e-5
    manifest=json.loads((OUT/'resonance_validation_manifest.json').read_text())
    assert manifest['windows']==688 and manifest['target_windows']==5203
    assert all(r['path'].endswith('_val.txt') for r in manifest['sources'])
    assert all(r['path'].endswith('_val.txt') for r in manifest['selected'])
    for r in manifest['sources']:check(r['path'],r['sha256'])
    assert not failed
    tests=ET.parse(OUT/'unit_tests.xml')
    assert len(tests.findall('.//testcase'))==449
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    decision=json.loads((OUT/'final_decision.json').read_text())
    assert decision['decision']=='NO-GO' and decision['recommended_backbone'] is None
    assert decision['no_third_candidate'] and decision['no_next_stage']
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    allowed=('configs/stage8/','scripts/stage8/','reports/stage8/','results/tables/stage8/','tests/test_stage8_admission_contracts.py')
    changed=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith(allowed) for p in changed)
    files=[]
    for directory in ('configs/stage8','scripts/stage8','reports/stage8','results/tables/stage8'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.append(ROOT/'tests/test_stage8_admission_contracts.py')
    patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[0-9A-Z]{16}']
    for path in files:
        assert path.stat().st_size<5_000_000
        assert path.suffix not in {'.pth','.pt','.ckpt','.pkl','.npy','.npz','.pdf','.pem','.key'}
        if path.suffix!='.gz':
            assert not any(re.search(pattern,path.read_bytes()) for pattern in patterns)
    for path in ('experiments/stage8/cache/Re/resonance/model.py',
                 'experiments/stage8/cache/Re_weights/rezara2/P8_haar_fixed_epoch129.pt',
                 'experiments/stage8/cache/venv/bin/python','experiments/stage8/cache/literature/resonance.pdf'):
        assert subprocess.run(['git','check-ignore','-q',path],cwd=ROOT).returncode==0
    assert len(list((ROOT/'reports/stage8').glob('*.md')))==8
    excluded={'delivery_checks.json','delivery_SHA256.csv'}
    csv_new(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted(files) if p.name not in excluded])
    json_new(OUT/'delivery_checks.json',dict(base_commit=BASE,
        protocol_commit='19c19d9',execution_head_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        final_commit='Recorded by final git log; no impossible self-containing commit SHA',
        decision='NO-GO',recommended_backbone=None,reports=8,full_tests_passed=449,
        historical_checks=count,historical_files_unchanged=True,
        scope='OriginalRe GPU engineering+MoFlowCPU component checks only; no formal epochs, no test metrics, no innovation module',
        full_regression_tests='May include existing synthetic forwards/backwards/optimizer regression tests; not research training',
        checked_sensitive_files=True,checked_ignore=True,diff_check=True,automatic_push=False))
    print('Passed:449 tests;8 reports; historical and new official artifact SHA checks',count)


if __name__=='__main__':main()
