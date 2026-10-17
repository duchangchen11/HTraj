"""Freeze Stage8 scope and extend historical SHA protection before probes."""
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
BASE='28a1c50482802ab0ea55a791b22cba56a8e5a6a4'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    with gzip.open(ROOT/'configs/stage7p6/protected_files.csv.gz','rt') as f:
        files={r['path']:r for r in csv.DictReader(f)}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for rel in filter(None,tracked):files.setdefault(rel,dict(path=rel,sha256=sha(ROOT/rel)))
    # Stage7.6 explicitly indexed its newly obtained ignored source/PDF files.
    with (ROOT/'results/tables/stage7p6/evidence_SHA256.csv').open() as f:
        for row in csv.DictReader(f):files.setdefault(row['path'],dict(path=row['path'],sha256=row['sha256']))
    csv_new(ROOT/'configs/stage8/protected_files.csv.gz',sorted(files.values(),key=lambda r:r['path']),fieldnames=['path','sha256'])
    old=json.loads((ROOT/'configs/stage7p6/protocol.json').read_text())
    json_new(ROOT/'configs/stage8/protocol.json',dict(
        base_commit=BASE,branch='feat/stage8-topic-and-backbone-selection',
        candidate_order=['cocoon2wong/Re','DSL-Lab/MoFlow'],maximum_candidate_models=2,
        literature_only_comparators=['AgentFormer','TUTR','SocialCircle','Inter-GraphFormer','direct recent related work'],
        historical_AgentFormer_route='Terminated; existing author-checkpoint reproduction retained as benchmark',
        standard_protocol='Fivefold ETH/UCY,8/12,0.4s,metres,minADE20/minFDE20 independently; different cohorts never ranked',
        formal_training=False,test_metrics_access=False,test_set_reevaluation=False,new_modules=False,
        hardware='RTX3080 10GiB',GPU_budget_seconds=7200,
        permitted_probes='Frozen original source; CPU structure/count; at most small synthetic/official train-or-validation forwards, original finite loss and one backward/update/checkpoint probe; not epochs',
        engineering_gate='Inspect protocol and specific falsifiable structural issue first; if neither problem nor fair protocol can be specified stop GPU probes',
        probe_seed=17,K=20,no_checkpoint_selection_by_test=True,
        limits='No third candidate, no new formal epoch, no test-driven hyperparameter changes, no truncation to hideOOM',
        literature='Primary paper methods and fixed author source; lack of full text stays unverified',
        GO='Concrete supported issue + distinct mathematical idea + legal train/validation + fair controls + feasible hardware/budget + explicit failure rule; otherwise NO-GO',
        protection_index='configs/stage8/protected_files.csv.gz',protection_index_SHA256=sha(ROOT/'configs/stage8/protected_files.csv.gz'),
        prediction_SHA_indices=old['prediction_SHA_indices'],automatic_push=False))
    print('Frozen Stage8 scope; protected file entries',len(files))


if __name__=='__main__':main()
