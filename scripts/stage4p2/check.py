"""Verify old-artifact seals, real result accounting and final local delivery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,gzip,json,subprocess,re,hashlib,argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage4p2');BASE='ce77c49c68367b169bc6d8f7a29bb4cba4400575'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='delivery_checks.json');args=parser.parse_args()
    protected=json.loads(Path('configs/stage4p2/protected_artifacts.json').read_text())['files']
    for row in protected:assert sha256(row['path'])==row['sha256'],row['path']
    config=json.loads(Path('configs/stage4p2/protocol.json').read_text())
    frozen=subprocess.check_output(['git','show','daf1e55:configs/stage4p2/protocol.json'])
    assert hashlib.sha256(frozen).hexdigest()==sha256('configs/stage4p2/protocol.json')
    assert sha256(config['cache'])==config['cache_sha256']
    counts={};hashes=set();summaries={}
    for fold in config['folds']:
        info=config['manifests'][fold];assert sha256(info['path'])==info['sha256']
        manifest=json.loads(Path(info['path']).read_text());train=manifest['splits']['train'];val=manifest['splits']['validation']
        assert {r['recording'] for r in train}.isdisjoint({r['recording'] for r in val})
        assert {r['recording'] for r in val}=={fold}
        for split in (train,val):
            for row in split:
                assert row['recording']!='biwi_eth' and len(row['frames'])==20
                assert all(b-a==10 for a,b in zip(row['frames'],row['frames'][1:]))
        counts[fold]=sum(len(r['agent_ids']) for r in val)
        for source in manifest['source_files'].values():assert sha256(source['path'])==source['sha256']
        for kind in ('G0','G1'):
            root=ROOT/fold/kind/'deterministic_scan';summary=json.loads((root/'summary.json').read_text());summaries[fold,kind]=summary
            run=json.loads((root/'run_config.json').read_text())
            for path,digest in run['code_files'].items():assert sha256(path)==digest
            assert sha256('scripts/stage4p2/train.py')==run['entrypoint_sha256']
            hashes.add(summary['initial_state_sha256']);curve=list(csv.DictReader((root/'curves.csv').open()))
            assert len(curve)==30 and [int(r['epoch']) for r in curve]==list(range(1,31))
            expected=min(curve,key=lambda r:(float(r['K1_distribution_mean_ADE']),int(r['epoch'])))
            assert int(expected['epoch'])==summary['best']['epoch'] and summary['last']['epoch']==30
            assert summary['parameters']==1553 and summary['smoke_passed']
            for label in ('best','last'):
                rows=list(csv.DictReader(gzip.open(root/(label+'_targets.csv.gz'),'rt')))
                assert len(rows)==3*counts[fold]
    assert len(hashes)==1
    inventory=list(csv.DictReader((ROOT/'checkpoint_inventory.csv').open()));assert len(inventory)==120
    for row in inventory:assert sha256(row['path'])==row['sha256']
    paired=list(csv.DictReader(gzip.open(ROOT/'paired_target_errors.csv.gz','rt')))
    assert len(paired)==sum(counts.values())*6
    assert len({(r['fold'],r['checkpoint_role'],r['window_id'],r['agent_id'],r['mode']) for r in paired})==len(paired)
    for row in paired:
        assert abs(float(row['G1_ADE_m'])-float(row['G0_ADE_m'])-float(row['delta_ADE_m']))<1e-12
    # Same-checkpoint empty-row predictions must have no graph-induced difference.
    bykey=defaultdict(dict)
    for r in csv.DictReader(gzip.open(ROOT/'checkpoint_intervention_targets.csv.gz','rt')):
        if 'legal_neighbors_none' in r['groups'].split('|'):
            bykey[r['fold'],r['window_id'],r['agent_id'],r['mode']][r['intervention']]=r
    checks=[]
    for fold in config['folds']:
        selected=[v for k,v in bykey.items() if k[0]==fold];differences=[]
        for triplet in selected:
            assert set(triplet)=={'real','zero','control'}
            differences.extend(abs(float(triplet[a][metric])-float(triplet['real'][metric])) for a in ('zero','control') for metric in ('ADE_m','FDE_m'))
        assert max(differences)==0
        checks.append(dict(fold=fold,target_windows=len(selected)//3,max_same_checkpoint_no_neighbor_error_difference=max(differences)))
    no_neighbor_path=ROOT/'no_neighbor_inference_check.csv'
    if no_neighbor_path.exists():
        old=list(csv.DictReader(no_neighbor_path.open()))
        assert len(old)==len(checks)
        for a,b in zip(old,checks):
            assert a['fold']==b['fold'] and int(a['target_windows'])==b['target_windows'] and float(a['max_same_checkpoint_no_neighbor_error_difference'])==b['max_same_checkpoint_no_neighbor_error_difference']
    else:csv_new(no_neighbor_path,checks)
    tree=ET.parse(ROOT/'final_tests.xml');tests=tree.findall('.//testcase');failures=tree.findall('.//failure');errors=tree.findall('.//error');skipped=tree.findall('.//skipped')
    assert len(tests)==237 and not failures and not errors and not skipped
    changes=subprocess.check_output(['git','diff','--name-status',BASE],text=True).splitlines()
    assert all(r.startswith('A\t') for r in changes),'Old tracked artifact modified: '+repr(changes)
    base_paths=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines())
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    added=sorted(set(p for p in paths if p and p not in base_paths));largest=[]
    for path in added:
        p=Path(path);assert p.stat().st_size<8*1024**2,(path,p.stat().st_size)
        assert p.suffix not in ('.pt','.pth','.ckpt','.npy','.npz','.pem','.key')
        assert not path.startswith(('data/raw/','data/processed/','third_party/'))
        if p.suffix!='.gz':assert not re.search(r'(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|BEGIN (RSA |OPENSSH )?PRIVATE KEY)',p.read_text(errors='ignore'))
        largest.append((p.stat().st_size,path))
    subprocess.run(['git','diff','--check'],check=True)
    decision=json.loads((ROOT/'first_round_decision.json').read_text());assert decision['run_G2'] is False and decision['additional_seeds_executed'] is False
    write_json_new(ROOT/args.output,dict(passed=True,protected_files=len(protected),old_hashes_unchanged=True,
        tests=len(tests),failures=0,errors=0,skipped=0,checkpoints_verified=len(inventory),paired_target_rows=len(paired),validation_target_windows=counts,
        initial_states_identical=True,protocol_unchanged_after_registration=True,training_source_commit=summaries[config['folds'][0],'G0']['source_commit'],
        training_code_matches_run_hashes=True,postprocessing_script_hashes={str(p):sha256(p) for p in sorted(Path('scripts/stage4p2').glob('*.py'))},
        base_commit=BASE,branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        new_files_scanned=len(added),largest_new_files=sorted(largest,reverse=True)[:10],no_data_weights_secrets=True,
        ETH_evaluated=False,G2_trained=False,additional_seeds_executed=False,push_executed=False,
        logs_note='execution_logs archive contains snapshots captured by analyze.py; its own then-active analysis.log was empty in that snapshot',
        final_log_hashes={str(p):sha256(p) for p in sorted(Path('experiments/stage4p2').glob('*.log'))}))
    print('PASS:',len(protected),'old assets unchanged;',len(tests),'tests;',len(inventory),'checkpoints;',len(paired),'paired rows')


if __name__=='__main__':main()
