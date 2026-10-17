"""Stage7.6: saved-evidence/static audit and analytic probability checks only.

No forecasting model is imported or executed. No prediction/test metric is read.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new

BASE = 'a00c9a38793135b77635ee4f99862f1f017eee23'
OFFICIAL = 'e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5'
OUT = ROOT / 'results/tables/stage7p6'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def supervised_sets(row):
    ids = json.loads(row['ids_json'])
    lengths = json.loads(row['future_lengths_json'])
    targets = json.loads(row['target_ids_json'])
    if len(set(ids)) != len(ids) or len(ids) != len(lengths):
        raise ValueError('Invalid context IDs or lengths')
    if not set(targets).issubset(ids):
        raise ValueError('S must be a subset of C')
    if any(lengths[ids.index(t)] != 12 for t in targets):
        raise ValueError('Supervised target does not have full labels')
    return ids, targets, lengths


def normal_kl(mu, variance):
    """KL N(mu,variance) || N(0,1), an analytic example, not a model."""
    if not variance > 0 or not math.isfinite(mu + variance):
        raise ValueError('Finite positive variance required')
    return .5 * (mu * mu + variance - 1 - math.log(variance))


def constrained_gaussian_bound(y, noise_variance):
    """p(z_s)p(z_u)N(y;z_s+z_u,noise); q_u=p_u.

    This toy example proves validity but also the looseness of prior completion.
    It is not implemented as an AgentFormer architecture or training loss.
    """
    if noise_variance <= 0:
        raise ValueError('Positive noise variance required')
    mu = y / (1 + noise_variance)
    variance = noise_variance / (1 + noise_variance)
    sq_error = (y - mu) ** 2 + variance + 1
    recon = -.5 * (math.log(2 * math.pi * noise_variance) + sq_error / noise_variance)
    kl_s = normal_kl(mu, variance)
    evidence = -.5 * (math.log(2 * math.pi * (2 + noise_variance)) + y * y / (2 + noise_variance))
    elbo = recon - kl_s
    return dict(y=y, noise_variance=noise_variance, q_s_mu=mu, q_s_variance=variance,
                KL_s=kl_s, KL_u=0., ELBO=elbo, log_evidence=evidence, gap=evidence-elbo,
                true_posterior_u_mu=y / (2 + noise_variance))


def supervised_average(values):
    if not values:
        raise ValueError('No supervised target: skip this training window')
    return sum(values) / len(values)


def prepare():
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != BASE:
        raise RuntimeError('Unexpected base revision')
    old = load('configs/stage7p5/protected_artifacts.json')
    # Extend the existing frozen inventory, never replace its expected hashes.
    files = {r['path']: r for r in old['files']}
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for rel in filter(None, tracked):
        files.setdefault(rel, dict(path=rel, sha256=digest(ROOT / rel)))
    for path in (ROOT / 'experiments/stage7p5/cache').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            rel = str(path.relative_to(ROOT))
            files.setdefault(rel, dict(path=rel, sha256=digest(path)))
    csv_new(ROOT / 'configs/stage7p6/protected_files.csv.gz', sorted(files.values(), key=lambda r:r['path']),
            fieldnames=['path', 'sha256'])
    json_new(ROOT / 'configs/stage7p6/protocol.json', dict(
        name='Stage7.6 static C/S feasibility audit v1', base_commit=BASE, official_commit=OFFICIAL,
        branch='feat/stage7p6-causal-context-research-decision',
        official_protocol='ETH/UCY fivefold complete20,8/12,0.4s,metres,DLow20,minADE/minFDE separately',
        new_training=False, new_model_inference=False, new_model_code=False, optimizer_updates=0,
        numerical_tests='CPU analytic probability/budget/evidence checks; not predictive performance',
        historical_test_metrics_access=False, prediction_files='Hash opaque bytes for preservation only',
        old_counterfactuals='Reuse Stage7.1 original raw-data tests, do not rerun or relabel as official-loader success',
        limitations='Stage7.1 cohort is Stage5.1 support, not official74-file benchmark prevalence',
        protection_index='configs/stage7p6/protected_files.csv.gz',
        protection_index_SHA256=digest(ROOT / 'configs/stage7p6/protected_files.csv.gz'),
        prediction_SHA_indices=old['prediction_SHA_indices'],
        literature_window='2022-2026 plus AgentFormer2021/Trajectron++2020 foundations',
        source_priority='Author/venue papers and fixed source; inaccessible methods remain unknown',
        decision='GO requires problem, novelty, consistent math, no fictitious GT, fair controls, resources, declared protocol all satisfied; otherwise NO-GO and stop this route',
        automatic_push=False))
    print('Prepared frozen protocol and', len(files), 'historical file hashes')


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT/'results/tables/stage7p1/context_windows.csv.gz', 'rt') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    hist = Counter()
    for row in rows:
        ids, targets, lengths = supervised_sets(row)
        groups[row['fold'], row['split']].append((row, ids, targets, lengths))
        for n in lengths:
            hist[row['fold'], row['split'], n] += 1
    summary = []
    for (fold, split), group in sorted(groups.items()):
        summary.append(dict(fold=fold, split=split, windows=len(group),
            context_node_windows=sum(len(i) for _,i,_,_ in group),
            supervised_target_windows=sum(len(s) for _,_,s,_ in group),
            incomplete_future_node_windows=sum(sum(n<12 for n in ns) for _,_,_,ns in group),
            no_future_node_windows=sum(ns.count(0) for _,_,_,ns in group),
            mixed_label_windows=sum(any(n<12 for n in ns) for _,_,_,ns in group),
            full_future_context_not_supervised=sum(sum(n==12 for n in ns)-len(s) for _,_,s,ns in group),
            partial_history_excluded=sum(int(row['partial_history_excluded']) for row,_,_,_ in group),
            max_context=max(len(i) for _,i,_,_ in group)))
    csv_new(OUT/'saved_cohort_summary.csv', summary)
    csv_new(OUT/'future_label_length_histogram.csv', [dict(fold=f, split=s, future_length=n, context_node_windows=v)
            for (f,s,n),v in sorted(hist.items())])
    cf = []
    for path in sorted((ROOT/'results/tables/stage7p1/counterfactuals').glob('*.json')):
        a = json.loads(path.read_text())
        checks = {k:v for k,v in a.items() if k.endswith('_equal')}
        raw = ROOT / a['raw_path']
        expected = a['raw_sha256']
        actual = digest(raw)
        if checks and not all(checks.values()):
            raise RuntimeError('Old counterfactual has failed check')
        if actual != expected:
            raise RuntimeError('Raw counterfactual SHA changed')
        cf.append(dict(source=str(path.relative_to(ROOT)), source_SHA256=digest(path),
                       raw_path=a['raw_path'], raw_SHA256=actual,
                       checks=checks, K1_max_difference=a['K1_prediction_max_difference'],
                       K20_max_difference=a['K20_prediction_max_difference']))
    if len(cf) != 10:
        raise RuntimeError('Expected 10 saved tests')
    json_new(OUT/'counterfactual_reuse.json', dict(new_inference=False, tests=cf,
        scope='Strict-history adapter prior tests, not proof official complete20 eligibility is causal'))
    probes = [load('results/tables/stage7p1/negative_probes/'+f+'.json')
              for f in ('biwi_hotel','crowds_zara02')]
    json_new(OUT/'negative_probe_reuse.json', dict(new_training=False, original_probes=probes,
        interpretation='Mask-only objective fails original mixed-label posterior. Not real forecasting metrics.'))
    cvae = load('results/tables/stage7p4/execution_v2/CVAE_epoch.json')
    dlow = load('results/tables/stage7p4/execution_v2/DLow_epoch.json')
    prev = load('results/tables/stage7p4/execution_v2/compute_budget.json')
    large = load('results/tables/stage7p1/maximum_context_resource.json')
    ratio = 73/57
    budget = dict(measured_reference='Stage7.4 original complete20 HOTEL, one epoch each',
        measured_CVAE=cvae['resource'], measured_DLow=dlow['resource'],
        original_CVAE100_DLow50_hours=(100*cvae['resource']['seconds']+50*dlow['resource']['seconds'])/3600,
        original_fivefold_hours={k:prev[k] for k in ('fivefold_point_hours','fivefold_low_hours','fivefold_high_hours')},
        N73_prior_inference_only=large['resource'], N73_training_tested=False,
        ratio_73_to_57=ratio, attention_ratio_squared=ratio**2,
        illustrative_linear_CVAE_allocated_bytes=cvae['resource']['peak_allocated_bytes']*ratio,
        illustrative_quadratic_CVAE_allocated_bytes=cvae['resource']['peak_allocated_bytes']*ratio**2,
        single_attention_matrix_FP32_bytes={str(n):4*20*8*(12*n)**2 for n in (57,73)},
        time_scenarios_same_window_count_hours=[(100*cvae['resource']['seconds']+50*dlow['resource']['seconds'])/3600*f for f in (1,ratio,ratio**2)],
        caveat='Scenarios, not measured73 training or rigorous bounds; new official causal context population unknown. Full model contains nonquadratic terms. Allocation != reserved != whole-card.',
        current_storage=dict(free_bytes=shutil.disk_usage(ROOT).free, training_started=False),
        permission='Budget only; no100+50epoch run authorized/executed')
    json_new(OUT/'resource_audit.json', budget)
    json_new(OUT/'analytic_checks.json', dict(kind='Analytic toy Gaussian, no forecast',
        examples=[constrained_gaussian_bound(y,v) for y,v in ((0.,1.),(2.,.5),(4.,.1))],
        no_model_import_or_execution=True))
    sources = ['configs/stage7p6/protocol.json', 'results/tables/stage7p1/context_windows.csv.gz',
        'results/tables/stage7p1/data_audit.json', 'results/tables/stage7p1/execution_summary.json',
        'results/tables/stage7p4/execution_v2/CVAE_epoch.json','results/tables/stage7p4/execution_v2/DLow_epoch.json',
        'results/tables/stage7p4/execution_v2/compute_budget.json',
        'configs/stage5p1/manifests/biwi_hotel/B.json','configs/stage5p1/manifests/crowds_zara02/B.json']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/stage7p6/cache/literature').rglob('*') if p.is_file()]
    sources += ['experiments/stage7/cache/AgentFormer/'+OFFICIAL+'/'+p for p in
        ('data/preprocessor.py','model/agentformer.py','model/agentformer_loss.py','model/dlow.py','model/agentformer_lib.py')]
    csv_new(OUT/'evidence_SHA256.csv', [dict(path=p, sha256=digest(ROOT/p), bytes=(ROOT/p).stat().st_size)
                                    for p in sorted(sources)])
    json_new(OUT/'execution_record.json', dict(base_commit=BASE,
        analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        official_commit=OFFICIAL, formal_training=False, new_model_forward=False, ETH_metrics_read=False,
        command='experiments/stage4/venv/bin/python scripts/stage7p6/audit.py analyze',
        source_sha256=digest(Path(__file__)), total_saved_windows=len(rows), evidence_only=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def verify():
    protocol = load('configs/stage7p6/protocol.json')
    failures = []
    count = 0
    def check(rel, expected):
        nonlocal count
        count += 1
        if not (ROOT/rel).is_file() or digest(ROOT/rel) != expected:
            failures.append(rel)
    check(protocol['protection_index'],protocol['protection_index_SHA256'])
    with gzip.open(ROOT/protocol['protection_index'], 'rt') as stream:
        for row in csv.DictReader(stream):
            check(row['path'],row['sha256'])
    for index in protocol['prediction_SHA_indices']:
        check(index['path'], index['sha256'])
        # Opaque historical predictions: hash bytes, never parse coordinates/GT/errors.
        with gzip.open(ROOT/index['path'], 'rt') as stream:
            for row in csv.DictReader(stream):
                check(row['path'], row.get('sha256',row.get('SHA256')))
    json_new(OUT/'historical_preservation.json', dict(checked_files=count, failures=failures,
        prediction_content_interpreted=False, passed=not failures))
    if failures:
        raise RuntimeError('Historical hashes changed')
    print('Historical SHA preservation passed:',count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','analyze','verify'))
    getattr(sys.modules[__name__], parser.parse_args().action)()
