import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from src.agentformer_benchmark.official import ROOT, SOURCE, DATA, install_compatibility

OUT = ROOT/'results/tables/stage7p3'
CACHE = ROOT/'experiments/stage7p3/cache'
WORK = CACHE/'workspace'
FOLDS = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
SEQUENCES = dict(eth=['biwi_eth'], hotel=['biwi_hotel'], univ=['students001', 'students003'],
                 zara1=['crowds_zara01'], zara2=['crowds_zara02'])


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def protocol():
    return json.loads((ROOT/'configs/stage7p3/evaluation_protocol.json').read_text())


def verify_inputs():
    p = protocol()
    rows = p['official_sources']+p['data_files']+p['checkpoints_and_configs']
    for r in rows:
        assert digest(ROOT/r['path']) == r['sha256'], r['path']
    for path, expected in p['execution_source_SHA256'].items():
        assert digest(ROOT/path) == expected, path
    return dict(verified=True, official_source_files=len(p['official_sources']), data_files=len(p['data_files']),
                checkpoint_and_config_files=len(p['checkpoints_and_configs']), execution_files=len(p['execution_source_SHA256']))


def committed_clean_gate():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    assert git('branch', '--show-current') == 'feat/stage7p3-agentformer-official-evaluation'
    assert not git('status', '--porcelain', '--untracked-files=all'), 'Uncommitted files: freeze before inference'
    commit = git('rev-parse', 'HEAD')
    assert commit != protocol()['base_commit'], 'Independent preregistration commit required'
    paths = ['configs/stage7p3/evaluation_protocol.json', 'configs/stage7p3/test_windows.csv.gz']
    paths += list(protocol()['execution_source_SHA256'])
    paths += [r['path'] for r in protocol()['checkpoints_and_configs'] if r['path'].startswith('configs/')]
    for path in paths:
        blob = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest(ROOT/path), path
    return commit


def workspace():
    WORK.mkdir(parents=True, exist_ok=True)
    links = {WORK/'cfg': ROOT/'configs/stage7p3/official', WORK/'datasets/eth_ucy': DATA}
    for case in protocol()['pairs']:
        for kind in ['CVAE', 'DLow']:
            source = Path(case[f'{kind}_checkpoint']).parent
            links[WORK/'results'/case[f'{kind}_config']/'models'] = source
    for destination, source in links.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=True)
        assert destination.resolve() == source.resolve()
    return WORK


def official_functions(namespace):
    tree = ast.parse((SOURCE/'test.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
             n.name in {'save_prediction', 'get_model_prediction', 'test_model'}]
    assert len(nodes) == 3
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE/'test.py'), 'exec'), namespace)
    return namespace


def evaluator():
    install_compatibility()
    spec = importlib.util.spec_from_file_location('stage7p3_original_eval', SOURCE/'eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_candidates(files, ids, frame, raw):
    """Independent dictionary join, not the author's align_gt; require exact 12xK coverage."""
    expected_names = [f'sample_{k:03d}.txt' for k in range(20)]
    if [Path(f).name for f in files] != expected_names:
        raise ValueError('Missing, duplicated or extra candidate files')
    future = list(range(frame+1, frame+13))
    expected = {(t, i) for i in ids for t in future}
    all_rows, predictions = [], []
    for f in files:
        rows = np.loadtxt(f, ndmin=2, dtype=np.float32)
        if rows.shape != (len(expected), 4) or not np.isfinite(rows).all():
            raise ValueError('Invalid prediction shape/values')
        if not np.equal(rows[:, :2], np.round(rows[:, :2])).all():
            raise ValueError('Nonintegral frame/ID')
        index = {(int(t), int(i)): (x, y) for t, i, x, y in rows}
        if set(index) != expected or len(index) != len(rows):
            raise ValueError('Candidate frame/ID mismatch')
        predictions.append(np.array([[index[t, i] for t in future] for i in ids]))
        all_rows.append(rows)
    gt_index = {(int(t), int(i)): (x, y) for t, i, x, y in raw}
    truth = np.array([[gt_index[t, i] for t in future] for i in ids], dtype=np.float32)
    return np.stack(predictions, axis=1), truth, np.stack(all_rows)


def close_to_author(value, author, absolute=.01, relative=.05):
    return abs(value-author) <= max(absolute, relative*author)+1e-12
