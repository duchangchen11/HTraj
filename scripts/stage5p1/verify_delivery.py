"""Read-only legacy, provenance, causal cohort and actual-result verification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import importlib.metadata
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from src.data.preprocessing import sha256,write_json_new
from src.socialvae_causal.data import NON_ETH,PROTOCOL,OFFLINE
from src.socialvae_causal.io import read_csv
from src.socialvae_causal.official import SOURCE,EXPECTED

OUT=Path('results/tables/stage5p1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');args=parser.parse_args()
    protected=json.load(open('configs/stage5p1/protected_artifacts.json'))
    mismatch=[r['path'] for r in protected['files'] if not Path(r['path']).exists() or sha256(r['path'])!=r['sha256']]
    assert not mismatch,mismatch
    assert all(sha256(SOURCE/f)==h for f,h in EXPECTED.items())
    data=json.load(open(OUT/'data_execution.json'));raw=json.load(open(OUT/'raw_counterfactual_summary.json'))
    coverage=json.load(open(OUT/'coverage_gate.json'));smoke=json.load(open(OUT/'smoke_summary.json'))
    conf=json.load(open(OUT/'smoke_config.json'));resource=json.load(open(OUT/'resource_summary.json'))
    for rec,source in data['raw_sources'].items():
        assert rec in NON_ETH and source['scene']!='ETH'
        assert sha256(Path('data/raw/eth_ucy')/source['file'])==source['sha256']
    old=json.load(open('configs/data_manifests/stage2_v1/fold_ETH.json'))
    trainkeys={(w['recording'],w['start_frame']) for w in old['splits']['train']}
    valkeys={(w['recording'],w['start_frame']) for w in old['splits']['validation']}
    for ref in data['manifests']:
        assert sha256(ref['path'])==ref['sha256']
        m=json.load(open(ref['path']));tr=m['splits']['train'];va=m['splits']['validation']
        assert m['protocol'] in [PROTOCOL,OFFLINE]
        assert not {w['recording'] for w in tr}&{w['recording'] for w in va}
        assert {w['recording'] for w in va}=={m['validation_recording']}
        assert all(w['recording'] in NON_ETH and w['end_frame']-w['start_frame']==190 for w in tr+va)
        assert all((w['recording'],w['start_frame']) in trainkeys and (w['recording'],w['start_frame']) not in valkeys for w in tr)
        assert all((w['recording'],w['start_frame']) in valkeys for w in va)
    for ref in data['target_manifests']:assert sha256(ref['path'])==ref['sha256']
    checks=read_csv(OUT/'raw_counterfactual_checks.csv')
    assert len(checks)==36 and all(r['passed']=='True' and float(r['B_prediction_max_difference_m'])==0. for r in checks)
    assert raw['passed'] and coverage['passed'] and resource['finite_gate']
    observations=json.load(open(OUT/'observability_index.json'));observed_rows=0
    for ref in observations['files']:
        assert sha256(ref['path'])==ref['sha256']
        for row in read_csv(ref['path']):
            p,v,a=(int(row[k]) for k in ['position_valid_bits','velocity_valid_bits','acceleration_valid_bits'])
            assert v==(p & (p<<1) & 254)
            assert a==(v & (v<<1) & 252)
            assert int(row['motion_attention_valid_bits'])==v
            observed_rows+=1
    assert observed_rows==58454
    for f,h in conf['source_sha256'].items():assert sha256(f)==h, f
    assert conf['actual_batch_size']==128 and conf['seed']==17
    assert conf['train_target_windows']==23545 and conf['validation_target_windows']==1092
    assert 'crowds_zara02' not in conf['train_recordings'] and conf['validation_recordings']==['crowds_zara02']
    steps=read_csv(OUT/'smoke_step_curves.csv.gz');curve=read_csv(OUT/'smoke_epoch_curves.csv')
    assert len(steps)==920 and len(curve)==5 and smoke['total_target_exposures']==117725
    assert all(math.isfinite(float(r[k])) for r in steps for k in ['loss','rec','kl','gradient_L2'])
    assert smoke['all_finite'] and smoke['all_strict_restore_passed'] and smoke['train_loss_decreased']
    best=min(curve,key=lambda r:(float(r['K1_prior_ADE_m']),int(r['epoch'])))
    assert int(best['epoch'])==smoke['best']['epoch']==5
    checkpoints=read_csv(OUT/'checkpoint_inventory.csv');assert len(checkpoints)==9
    for r in checkpoints:
        assert r['path'].startswith('experiments/stage5p1/checkpoints/')
        assert sha256(r['path'])==r['sha256']
        assert subprocess.run(['git','check-ignore','-q',r['path']]).returncode==0
    assert all(r['status']=='passed' for r in resource['rows']) and len(resource['rows'])==4
    assert sum(r['optimizer_steps'] for r in resource['rows'])==52
    cv=read_csv(OUT/'cv_paired_targets.csv.gz');paired=read_csv(OUT/'smoke_cv_paired_best.csv.gz')
    assert len(cv)==4230 and len(paired)==3276
    for fold,count in [('biwi_hotel',318),('crowds_zara02',1092)]:
        manifest=read_csv(OUT/'manifests'/fold/'B_validation_targets.csv.gz')
        truth={(r['recording'],r['start_frame'],r['target_id']):r['future_sha256'] for r in manifest}
        assert len(truth)==count
        for model in ['CV-Last2','CV-OLS4','CV-OLS8']:
            rows=[r for r in cv if r['fold']==fold and r['model']==model]
            assert len(rows)==count
            assert all(truth[(r['recording'],r['start_frame'],r['target_id'])]==r['future_sha256'] for r in rows)
    assert all(float(r['neural_K1_ADE_m'])-float(r['CV_ADE_m'])==float(r['delta_ADE_m']) for r in paired)
    suites=ET.parse(OUT/'unit_tests.xml').getroot().findall('testsuite')
    tests=sum(int(s.attrib['tests']) for s in suites);failures=sum(int(s.attrib['errors'])+int(s.attrib['failures']) for s in suites)
    assert tests>=293 and failures==0
    allowed=('configs/stage5p1/','reports/stage5p1/','results/tables/stage5p1/','scripts/stage5p1/','src/socialvae_causal/','experiments/stage5p1/')
    changed=set(subprocess.check_output(['git','diff','--name-only',protected['base_commit']],text=True).splitlines())
    changed |= set(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines())
    assert all(p.startswith(allowed) or p=='tests/test_socialvae_causal.py' for p in changed)
    assert not any(Path(p).suffix in {'.pth','.pt','.npz','.npy','.ckpt','.pkl','.safetensors'} for p in changed)
    assert all(Path(p).stat().st_size<20_000_000 for p in changed)
    subprocess.run(['git','diff','--check'],check=True)
    decision=json.load(open(OUT/'research_decision.json'))
    assert decision['engineering_ready_for_formal_baseline'] and not decision['social_information_independent_benefit_proven']
    result=dict(protected_files=len(protected['files']),legacy_mismatches=mismatch,official_source_unchanged=True,
        raw_counterfactual_cases=36,raw_counterfactual_max_difference_m=0.,observable_actor_windows=observed_rows,
        tests=tests,test_failures=failures,training_smoke_epochs=5,training_smoke_updates=920,resource_updates=52,
        actual_batch_sizes=[r['batch_size'] for r in resource['rows']],new_checkpoint_files=len(checkpoints),
        checkpoints_git_ignored=True,old_validation_promoted_to_train=False,ETH_loaded=False,
        engineering_ready_for_formal_baseline=True,formal_baseline_training_completed=False,new_architecture=False,automatic_push=False,
        git_diff_check=True,git_HEAD_at_check=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        packages={k:importlib.metadata.version(k) for k in ['torch','numpy','pytest']},
        stage5p1_code_sha256={str(f):sha256(f) for folder in ['scripts/stage5p1','src/socialvae_causal'] for f in sorted(Path(folder).glob('*.py'))})
    if args.record:write_json_new(OUT/'delivery_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
