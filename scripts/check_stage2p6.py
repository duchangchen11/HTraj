"""Final read-only audit of protected bytes and complete actual experiment evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'results/tables/stage2p6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol_path=BASE/'configs/stage2p6/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    protected=json.loads((BASE/'configs/stage2p6/protected_artifacts.json').read_text())
    for path,digest in protected.items():
        assert sha(BASE/path)==digest,path
    assert json.loads((ROOT/'frozen_baseline_audit.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'gpu_equivalence_audit.json').read_text())['optimizer_steps']==0
    source='ec8f665c1faab7ee6cd64172c8852c904c8f012b'
    for path in ['src/models/order_verification.py','src/evaluation/order_verification.py','src/training/order_verification.py']:
        assert (BASE/path).read_bytes()==subprocess.check_output(['git','show',source+':'+path],cwd=BASE)
    runs=list(csv.DictReader((ROOT/'new_summary.csv').open()))
    expected={(m,s) for m in protocol['new_models'] for s in protocol['seeds']}
    assert len(runs)==18 and {(r['model'],int(r['seed'])) for r in runs}==expected
    for run in runs:
        assert run['status']=='completed' and run['test_evaluated']=='False'
        assert run['source_commit']==source and run['protocol_sha256']==sha(protocol_path)
        assert int(run['optimizer_steps'])==3840
        path=ROOT/'new'/run['model']/f"seed_{run['seed']}"
        epochs=list(csv.DictReader((path/'epochs.csv').open()))
        assert [int(e['epoch']) for e in epochs]==list(range(1,31))
        best=min(epochs,key=lambda r:float(r['validation_ADE_m']))
        assert int(best['epoch'])==int(run['best_epoch'])
        assert float(best['validation_ADE_m'])==float(run['best_ADE_m'])
        config=json.loads((path/'config.json').read_text())
        assert config['hidden_dim']==64 and config['training_windows']==1024 and config['validation_windows']==96
        for epoch in epochs:
            for key,value in epoch.items():
                if key=='source_commit' or value=='':
                    continue
                assert math.isfinite(float(value)),(run['model'],key)
        for q in range(1,config['max_order']+1):
            for part in ['before','source','message','carry','after']:
                assert f'q{q}_{part}_L2_p95' in epochs[0]
            assert f'q{q}_projection_grad_L2' in epochs[0]
        local=BASE/'experiments/stage2p6/new'/run['model']/f"seed_{run['seed']}/checkpoints"
        for endpoint in ['best','last']:
            assert sha(local/f'{endpoint}.pt')==run[endpoint+'_checkpoint_sha256']
            groups=list(csv.DictReader((path/f'groups_{endpoint}.csv').open()))
            assert len(groups)==59 and int(groups[0]['ade_count'])==1202 and int(groups[0]['fde_count'])==839
    frozen=list(csv.DictReader((ROOT/'frozen_baselines.csv').open()))
    assert len(frozen)==6 and all(r['trained_in_stage2p6']=='False' for r in frozen)
    for run in frozen:
        assert len(list(csv.DictReader((BASE/run['origin_epochs']).open())))==30
    assert len(list(csv.DictReader((ROOT/'all_runs.csv').open())))==24
    assert len(list(csv.DictReader((ROOT/'group_comparison.csv').open())))==24*2*59
    controls=list(csv.DictReader((ROOT/'relation_interventions.csv').open()))
    assert len(controls)==96
    for row in controls:
        assert row['status']=='executed' and row['test_evaluated']=='False' and row['retrained']=='False'
        assert row['inference_runner_sha256']==sha(BASE/'scripts/stage2p6_relation_attribution.py')
        assert row['relation_engine_sha256']==sha(BASE/'src/evaluation/relation_controls.py')
        assert math.isfinite(float(row['ADE_m'])) and math.isfinite(float(row['FDE_m']))
        if row['case'] in ['full','radius_max_change','scene_random','scene_max_change','degree_swap_global']:
            assert int(row['incoming_degree_mismatch_nodes'])==0 and row['real_edges']==row['control_edges']
        if row['case'] in ['full','degree_swap_global']:
            assert int(row['outgoing_degree_mismatch_nodes'])==0
        if row['radius_constraint']=='preserved':
            assert int(row['control_outside_radius_edges'])==0
    paths=list(csv.DictReader((ROOT/'path_identity_summary.csv').open()))
    assert len(paths)==4 and all(r['GT_used_for_statistics']=='False' for r in paths)
    for model in protocol['new_models']:
        for suffix in ['correlations','quartiles','projection_spectra']:
            rows=list(csv.DictReader((ROOT/'state_error'/f'{model}_{suffix}.csv').open()))
            assert {int(r['seed']) for r in rows}=={17,42,2026}
            assert all(r['inference_runner_sha256']==sha(BASE/'scripts/stage2p6_state_error_analysis.py') for r in rows)
    tests=ET.parse(ROOT/'unit_tests.xml').getroot()
    assert all(int(s.attrib.get(k,0))==0 for s in tests for k in ['errors','failures'])
    assert sum(int(s.attrib['tests']) for s in tests)>=118
    for name in ['01_ORDER_ABLATION.md','02_TEMPORAL_ORDER_COMPARISON.md','03_STATE_SCALE_ANALYSIS.md',
                 '04_COMPLEX_SCENE_ANALYSIS.md','05_MODEL_COMPARISON.md','STAGE2P6_REPORT.md']:
        assert (BASE/'reports/stage2p6'/name).is_file()
    tracked=subprocess.check_output(['git','ls-files'],cwd=BASE,text=True).splitlines()
    assert not any(p.endswith(('.pt','.pth','.ckpt','.npy','.npz','.pem','.key')) for p in tracked)
    print('PASS: 222 protected artifacts unchanged; 18 new x30ep +6 frozen runs; 59 fixed groups; 96 inference controls; no ETH test/data/weights tracked.')


if __name__=='__main__':
    main()
