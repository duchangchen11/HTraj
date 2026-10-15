"""Read-only artifact checks; optionally write a new, non-overwriting audit record."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,gzip,json,subprocess,collections,re,xml.etree.ElementTree as ET
import numpy as np
from src.data.preprocessing import sha256,write_json_new
from src.models.social_diagnostics_models import DynamicResidualNormalized

ROOT=Path('results/tables/stage4')


def main():
    base='f12146143514d2b6eb6cf14877cb5b4cae454543'
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='feat/stage4-public-baseline'
    subprocess.run(['git','merge-base','--is-ancestor',base,'HEAD'],check=True)
    protection=json.loads(Path('configs/stage4/protected_artifacts.json').read_text())
    for entry in protection['files']:
        assert Path(entry['path']).is_file(),entry['path']
        assert sha256(entry['path'])==entry['sha256'],entry['path']
    sources=json.loads(Path('configs/stage4/official_sources.json').read_text())
    for source in sources['sources']:
        for entry in source['files']+source['assets']:
            assert sha256(entry['local'])==entry['sha256'],entry['local']
        for entry in source['assets']:
            path=entry['official_path'].lower()
            assert '/eth/' not in path and 'eth_ckpt' not in path and 'social-stgcnn-eth/' not in path,path
    social=json.loads((ROOT/'social_execution.json').read_text())
    assert not social['ETH_evaluated'] and not social['training'] and not social['paper_full_reproduction']
    assert social['protocol_sha256']==sha256('configs/stage4/protocol.json')
    assert social['official_sources_sha256']==sha256('configs/stage4/official_sources.json')
    assert social['script_sha256']==sha256('scripts/evaluate_stage4_social.py')
    for path,digest in social['source_hashes'].items():assert sha256(path)==digest
    results=list(csv.DictReader((ROOT/'social_results.csv').open()))
    assert len(results)==36
    counts=collections.Counter();sums=collections.defaultdict(lambda:np.zeros(2));banks={}
    with gzip.open(ROOT/'social_target_errors.csv.gz','rt') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['fold','monte_carlo_seed','window_index','agent_index','mode','ADE_m','FDE_m']
        for r in reader:
            assert r['fold'] in ('hotel','univ','zara1','zara2')
            a,f=float(r['ADE_m']),float(r['FDE_m']);assert np.isfinite([a,f]).all() and min(a,f)>=0
            key=(r['fold'],r['monte_carlo_seed'],r['mode']);counts[key]+=1;sums[key]+=[a,f]
            unit=(r['fold'],r['monte_carlo_seed'],r['window_index'],r['agent_index'])
            banks.setdefault(unit,{})[r['mode']]=np.array([a,f])
    for r in results:
        key=(r['fold'],r['monte_carlo_seed'],r['mode'])
        assert counts[key]==int(r['target_windows'])
        np.testing.assert_allclose(sums[key]/counts[key],[float(r['ADE_m']),float(r['FDE_m'])],atol=1e-10,rtol=0)
        assert r['official_function_numeric_match']=='True'
    for modes in banks.values():
        assert set(modes)=={'K1_first_sample','K1_distribution_mean','K20_independent_min'}
        assert np.all(modes['K20_independent_min']<=modes['K1_first_sample']+1e-12)
    temporal=list(csv.DictReader((ROOT/'social_temporal_support_audit.csv').open()))
    assert len(temporal)==5 and sum(int(r['target_windows']) for r in temporal)==33473
    assert all(int(r['windows_with_non10_frame_gap'])==0 for r in temporal)
    eq=json.loads((ROOT/'eqmotion_execution.json').read_text())
    assert eq['status']=='blocked_checkpoint_source_mismatch' and eq['actual_evaluated_targets']==0
    assert not eq['training'] and not eq['ETH_evaluated']
    for r in eq['diagnostics']:
        assert not r['strict_load_pass'] and r['ADE_m'] is None and r['FDE_m'] is None
    assert not (ROOT/'eqmotion_functional_results.csv').exists()
    geometry=list(csv.DictReader((ROOT/'eqmotion_synthetic_geometry_checks.csv').open()))
    assert len(geometry)==4 and all(r['pass_tolerance_1e_minus3']=='True' for r in geometry)
    tests=ET.parse(ROOT/'final_tests.xml').getroot().find('testsuite')
    assert tests is not None and int(tests.attrib['tests'])==210
    assert int(tests.attrib['failures'])==int(tests.attrib['errors'])==int(tests.attrib['skipped'])==0
    reports=['01_SOCIAL_STGCNN_AUDIT.md','02_SOCIAL_STGCNN_REPRODUCTION.md','03_EQMOTION_AUDIT.md',
        '04_PROTOCOL_COMPARISON.md','05_BASELINE_SELECTION.md','STAGE4_REPORT.md']
    assert all((Path('reports/stage4')/r).stat().st_size>1000 for r in reports)
    # After intentional git add, ensure every addition is reviewable and no old file changed.
    changes=subprocess.check_output(['git','diff','--name-status',base],text=True).splitlines()
    assert changes and all(r.startswith('A\t') for r in changes),changes
    paths=[r.split('\t',1)[1] for r in changes]
    private=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}')
    for name in paths:
        assert not name.startswith(('data/raw/','data/processed/','data/external/','experiments/stage4/venv/'))
        assert not any(part in name.lower() for part in ('.pth','.ckpt','.pkl','.npy','.npz','.pem'))
        path=Path(name);assert path.stat().st_size<8*1024*1024,(name,path.stat().st_size)
        if path.suffix!='.gz':assert not private.search(path.read_text()),name
    subprocess.run(['git','diff','--check',base],check=True)
    ignores=['data/external/stage4/assets/social_stgcnn/checkpoint/social-stgcnn-hotel/val_best.pth',
        'data/external/stage4/assets/eqmotion/eth_ucy/saved_models/hotel_ckpt.pth.tar',
        'experiments/stage4/venv/bin/python','data/raw/example.txt','data/processed/cache.npy','.env','credentials.json']
    for path in ignores:subprocess.run(['git','check-ignore','-q',path],check=True)
    artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='delivery_checks.json']
    logs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(Path('experiments/stage4').glob('*.log'))]
    record=dict(status='pass',base_commit=base,checked_at_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        verifier_sha256=sha256(__file__),old_protected_files=len(protection['files']),old_checkpoints=protection['old_checkpoints'],
        upstream_bytes_unchanged=True,old_results_unchanged=True,ETH_evaluated=False,new_training=False,
        full_unit_tests=210,unit_test_seconds=float(tests.attrib['time']),social_result_rows=len(results),social_target_error_rows=sum(counts.values()),
        independent_target_sampling_claim=False,eqmotion_checkpoint_status='blocked; no ADE/FDE',synthetic_geometric_checks=4,
        Dynamic_GNN_RN_hidden64_parameters=sum(p.numel() for p in DynamicResidualNormalized(hidden_dim=64).parameters()),
        added_files_checked=len(paths),largest_added_bytes=max(Path(p).stat().st_size for p in paths),
        whitespace_check=True,ignore_examples_pass=True,sensitive_pattern_matches=0,artifacts=artifacts,logs=logs,
        auto_push=False,limitations='secret check is specified-pattern scan; independent data confirmation and full paper/training reproduction not established')
    if '--record' in sys.argv:write_json_new(ROOT/'delivery_checks.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('artifacts','logs')}))


if __name__=='__main__':main()
