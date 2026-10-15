"""Final immutable-artifact, continuation, paired-target and stress-test checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,gzip,json,subprocess,hashlib,re,math
from collections import defaultdict
import xml.etree.ElementTree as ET
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
ROOT=Path('results/tables/stage4p2_final');BASE='55515d0ef5d50c36f4f148206d07f062644f92ff'


def main():
    config=json.loads(Path('configs/stage4p2_final/protocol.json').read_text());protected=json.loads(Path('configs/stage4p2_final/protected_artifacts.json').read_text())['files']
    for r in protected:assert sha256(r['path'])==r['sha256'],r['path']
    assert hashlib.sha256(subprocess.check_output(['git','show','930f92d:configs/stage4p2_final/protocol.json'])).hexdigest()==sha256('configs/stage4p2_final/protocol.json')
    assert sha256(config['cache'])==config['cache_sha256']
    for info in config['manifests'].values():assert sha256(info['path'])==info['sha256']
    inventory=[]
    for fold in config['folds']:
        for kind in ('G0','G1'):
            root=ROOT/'extension'/fold/kind;s=json.loads((root/'summary.json').read_text());run=json.loads((root/'run_config.json').read_text())
            assert s['completed'] and s['last']['epoch']==100 and s['parameters']==1553 and s['seed']==17
            assert json.loads((root/'restore_check.json').read_text())['passed']
            rows=list(csv.DictReader((root/'curve_31to100.csv').open()));assert [int(r['epoch']) for r in rows]==list(range(31,101))
            assert run['audit_helper_sha256']==sha256('src/evaluation/graph_final_audit.py')
            assert run['script_sha256']==sha256('scripts/stage4p2_final/extend.py')
            for p,digest in run['model_code'].items():assert sha256(p)==digest
            for r in rows:
                assert sha256(r['checkpoint'])==r['checkpoint_sha256']
                inventory.append(dict(path=r['checkpoint'],sha256=r['checkpoint_sha256'],bytes=Path(r['checkpoint']).stat().st_size))
    assert len(inventory)==280;csv_new(ROOT/'checkpoint_inventory.csv',inventory)
    cv=list(csv.DictReader(gzip.open(ROOT/'fresh_cv_targets.csv.gz','rt')));keys={(r['fold'],r['window_id'],r['agent_id']):r for r in cv};assert len(keys)==6886
    for role in ('budget30_best','budget100_best','budget100_last'):
        rows=list(csv.DictReader(gzip.open(ROOT/(role+'_paired_targets.csv.gz'),'rt')))
        assert len(rows)==6886 and {(r['fold'],r['window_id'],r['agent_id']) for r in rows}==set(keys)
        for r in rows:
            ref=keys[r['fold'],r['window_id'],r['agent_id']];assert r['future_sha256']==ref['future_sha256'] and r['history_sha256']==ref['history_sha256']
            for kind in ('CV-Last2','CV-OLS4','CV-OLS8'):
                for metric in ('ADE_m','FDE_m'):assert r[kind+'_'+metric]==ref[kind+'_'+metric]
            for kind in ('G0','G1','CV-Last2','CV-OLS4','CV-OLS8'):
                assert math.isfinite(float(r[kind+'_ADE_m'])) and math.isfinite(float(r[kind+'_FDE_m']))
    stress=json.loads((ROOT/'ptn_execution.json').read_text());assert stress['raw_rows']==330528 and not stress['future_GT_used'] and not stress['epsilon_modified']
    counts=0;g0_consistency=0;invalid=0
    for path in sorted((ROOT/'ptn').rglob('*.csv.gz')):
        rows=list(csv.DictReader(gzip.open(path,'rt')));counts+=len(rows);invalid+=sum(r['gaussian_valid']=='False' for r in rows)
        if path.name.startswith('G0_fixed'):
            counterpart=path.with_name(path.name.replace('_fixed_history_graph_','_recomputed_history_graph_'))
            other=list(csv.DictReader(gzip.open(counterpart,'rt')));assert len(rows)==len(other)
            for a,b in zip(rows,other):
                assert a['window_id']==b['window_id'] and a['agent_id']==b['agent_id'] and a['replicate']==b['replicate']
                assert a['final_relative_displacement_shift_m']==b['final_relative_displacement_shift_m']
            g0_consistency+=len(rows)
        for r in rows:
            assert math.isfinite(float(r['final_gain'])) and float(r['input_rms_m'])>0
            assert (r['high_risk']=='True')==(float(r['final_relative_displacement_shift_m'])>.1 and float(r['final_gain'])>100)
            assert float(r['CV_Last2_final_relative_shift_m'])<=24*float(r['epsilon_m'])+2e-4
    assert counts==330528 and g0_consistency==82632
    test=ET.parse(ROOT/'final_tests.xml');cases=test.findall('.//testcase')
    assert len(cases)==244 and not test.findall('.//failure') and not test.findall('.//error') and not test.findall('.//skipped')
    changes=subprocess.check_output(['git','diff','--name-status',BASE],text=True).splitlines();assert all(r.startswith('A\t') for r in changes)
    base_paths=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines())
    paths=[p for p in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0') if p and p not in base_paths]
    token=re.compile(r'(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|BEGIN (RSA |OPENSSH )?PRIVATE KEY)')
    largest=[]
    for p in paths:
        q=Path(p);assert q.stat().st_size<8*1024**2,(p,q.stat().st_size);assert q.suffix not in ('.pt','.pth','.ckpt','.npy','.npz','.pem','.key')
        assert not p.startswith(('data/','third_party/'))
        content=gzip.open(q,'rt').read() if q.suffix=='.gz' else q.read_text(errors='ignore')
        assert not token.search(content),p;largest.append((q.stat().st_size,p))
    subprocess.run(['git','diff','--check'],check=True)
    log_names=['cv.log','compare.log','ptn.log','final_tests.log']+[f'extension_{f}_{k}.log' for f in config['folds'] for k in ('G0','G1')]
    logs=[]
    with gzip.open(ROOT/'execution_logs.txt.gz','xt') as out:
        for name in log_names:
            p=Path('experiments/stage4p2_final')/name;out.write('\nFILE '+str(p)+'\n'+p.read_text());logs.append(dict(path=str(p),sha256=sha256(p)))
    write_json_new(ROOT/'delivery_checks.json',dict(passed=True,base_commit=BASE,branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        protected_files=len(protected),old_hashes_unchanged=True,protocol_unchanged=True,checkpoint_count=280,tests=len(cases),failures=0,errors=0,
        validation_target_windows=6886,paired_roles=3,PTN_raw_rows=counts,G0_fixed_vs_recomputed_matched_rows=g0_consistency,PTN_invalid_gaussian_rows=invalid,
        data_sha256=sha256(config['cache']),target_identity_sha256=sha256(ROOT/'target_identity_hashes.csv'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_source_commit='930f92d',
        script_hashes={str(p):sha256(p) for p in sorted(Path('scripts/stage4p2_final').glob('*.py'))},logs=logs,largest_new_files=sorted(largest,reverse=True)[:10],
        no_data_weights_secrets=True,ETH_evaluated=False,push_executed=False,network_modified=False,loss_modified=False,PTN_epsilon_modified=False))
    print('PASS:',len(protected),'protected files;',len(cases),'tests;',len(inventory),'new checkpoints;',counts,'PTN rows')


if __name__=='__main__':main()
