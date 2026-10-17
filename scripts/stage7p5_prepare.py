"""Freeze this read-only study before calculating any new GT errors."""
import hashlib
import json
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agentformer_innovation.metrics import fixed_subsets
from src.socialvae_baseline.artifacts import json_new


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    old=json.loads((ROOT/'configs/stage7p4/protected_artifacts.json').read_text())
    files={r['path']:r for r in old['files']}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for p in tracked:
        if p: files.setdefault(p,dict(path=p,sha256=digest(ROOT/p)))
    for p in (ROOT/'experiments/stage7p4/cache').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            rel=str(p.relative_to(ROOT));files.setdefault(rel,dict(path=rel,sha256=digest(p)))
    json_new(ROOT/'configs/stage7p5/protected_artifacts.json',dict(base_commit='50cbb13b8f2d4eff84eccf4c25025398b4ef3ed2',
        files=sorted(files.values(),key=lambda r:r['path']),prediction_SHA_indices=old['prediction_SHA_indices']))
    p4=json.loads((ROOT/'configs/stage7p4/protocol.json').read_text())
    indices=[f'results/tables/stage7p4/execution_v2/{f}_validation_prediction_SHA256.csv.gz' for f in ('hotel','zara2')]
    json_new(ROOT/'configs/stage7p5/protocol.json',dict(
        name='Stage7.5 frozen validation-only diagnosis v1',base_commit='50cbb13b8f2d4eff84eccf4c25025398b4ef3ed2',
        official_commit=p4['official_commit'],author_pairs=p4['author_pairs'],frozen_inputs=p4['frozen_inputs'],
        prediction_indices=[dict(path=p,sha256=digest(ROOT/p)) for p in indices],
        data_protocol='Official offline complete20, 8 history /12 future /0.4s /metres; source validation only',
        folds=['hotel','zara2'],test_metrics_access=False,model_updates=0,new_inference=False,
        groups='Only historical mean step speed <0.05, [0.05,0.4), [0.4,1.8), >=1.8m/s; historical OLS4 first/last direction >=30deg when >=0.05m/s; full8 historical neighbours within2m',
        subsets=fixed_subsets(),K=[1,2,5,10,20],subset_method='Nested prefix of original order and 20 fixed random permutations, seed7501, generated before GT errors',
        near_duplicate='Pair mean path distance <0.2m AND endpoint <0.3m; sensitivity half/double thresholds; not a calibrated mode count',
        motion='Path speed includes last observation->first prediction. Direction from last observation->endpoint, defined if displacement>=0.24m; OLS4 history heading>=0.05m/s',
        scale_control='ADE / max(true future path length,0.5m), sensitivities0.25/1m; FDE/net displacement similarly; ADE/history-implied12step travel floor0.5m. Continuous log scale regression with source fixed effects, descriptive only',
        prediction_summaries=['head0 original first candidate','all20 equal-head diagnostics, not probability-weighted expectation','ADE oracle','FDE oracle'],
        oracle='GT-dependent head winners are diagnostic bounds, never deployed selectors; winner bias explicitly reported',
        aggregation='Window-weighted, equal source-person ID, separate source recordings; overlapping folds are dependent. No window-level independent p-values',
        bootstrap='200 source-stratified person-cluster resamples, seed7502; descriptive interval, few recordings and selection exposure limit inference',
        social_analysis='History-only descriptive associations; no causal attribution or neighbour-intervention evidence',
        literature='Author papers/methods/experiments and official code; missing full text stays unverified. Primary identity checks 2023-2026, plus2021 foundation',
        decision='GO only supported scale-robust concrete defect + not fully covered mechanism + minimal feasible change + falsifiable ablations; otherwise NO-GO',
        training=False,automatic_push=False))


if __name__=='__main__': main()
