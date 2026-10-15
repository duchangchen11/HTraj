"""Read-only Stage5 provenance and history checks; --record writes once."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import json
import subprocess
import xml.etree.ElementTree as ET
from scripts.stage5.common import ROOT, OUT, sha, new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    protected = json.loads((ROOT/'configs/stage5/protected_artifacts.json').read_text())
    bad = [r['path'] for r in protected['files'] if not (ROOT/r['path']).exists() or sha(ROOT/r['path']) != r['sha256']]
    assert not bad, bad
    source = json.loads((OUT/'fetch_verification.json').read_text())['fetched_or_verified']
    assert all(sha(ROOT/r['local_path']) == r['sha256'] for r in source)
    run = json.loads((OUT/'socialvae_execution.json').read_text())
    assert sha(ROOT/'scripts/stage5/validate_socialvae.py') == run['script_sha256']
    assert sha(ROOT/'configs/stage5/functional_plan.json') == run['plan_sha256']
    eq = json.loads((OUT/'eqmotion_strict_loading.json').read_text())
    assert len(eq['configs']) == 2 and eq['unique_ETH_UCY_implementations'] == 1
    assert all(not c['strict_passed'] and not c['forward_executed'] for c in eq['configs'])
    assert all(sha(ROOT/c['checkpoint_path']) == c['checkpoint_sha256'] for c in eq['configs'])
    train = json.loads((OUT/'socialvae_training_smoke.json').read_text())
    assert train['steps'] == 1 and train['strict_restore'] and not train['full_training_completed']
    assert sha(ROOT/'experiments/stage5/assets/socialvae/smoke_seed17_one_step.pth') == train['checkpoint_sha256']
    assert not run['formal_training'] and not run['ETH_evaluated'] and not run['paper_reproduced']
    targets = list(csv.DictReader((OUT/'socialvae_target_errors.csv').open()))
    scores = list(csv.DictReader((OUT/'socialvae_metrics.csv').open()))
    for fold in ['hotel', 'zara02']:
        rows = [r for r in targets if r['fold'] == fold]
        assert len(rows) == 128
        assert len({(r['first_frame'],r['target_id']) for r in rows}) == len(rows)
        for mode, a, f in [('K1_prior_mean','K1_prior_mean_ADE_m','K1_prior_mean_FDE_m'),
                           ('K1_first_sample','K1_sample_ADE_m','K1_sample_FDE_m'),
                           ('K20_noFPC','K20_minADE_m','K20_minFDE_m')]:
            score = next(r for r in scores if r['fold']==fold and r['mode']==mode)
            assert abs(sum(float(r[a]) for r in rows)/128-float(score['ADE_m'])) < 1e-9
            assert abs(sum(float(r[f]) for r in rows)/128-float(score['FDE_m'])) < 1e-9
        assert all(float(r['K20_minADE_m']) <= float(r['K1_sample_ADE_m']) for r in rows)
        assert all(float(r['K20_minFDE_m']) <= float(r['K1_sample_FDE_m']) for r in rows)
    causality = json.loads((OUT/'socialvae_raw_loader_causality.json').read_text())
    assert causality['audit_passed'] and not causality['deployment_causality_passed']
    assert causality['forecast_mean_difference_m'] > 0
    selection = json.loads((ROOT/'configs/stage5/baseline_selection.json').read_text())
    assert selection['primary']['model'] == 'SocialVAE' and selection['backup']['model'] == 'SGCN'
    assert not selection['new_method_design_ready']
    suites = ET.parse(OUT/'unit_tests.xml').getroot().findall('testsuite')
    tests = sum(int(s.attrib['tests']) for s in suites)
    failures = sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    assert tests == 255 and failures == 0
    # New code and scientific records must stay confined to independent paths.
    base = protected['base_commit']
    names = set(subprocess.check_output(['git','diff','--name-only',base],text=True).splitlines())
    names |= set(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines())
    permitted = ('configs/stage5/','reports/stage5/','results/tables/stage5/','scripts/stage5/','experiments/stage5/')
    assert all(p.startswith(permitted) or p == 'tests/test_stage5_public_selection.py' for p in names), sorted(names)
    forbidden = {'.pt','.pth','.ckpt','.npz','.npy','.pkl','.safetensors'}
    assert not any(Path(p).suffix in forbidden for p in names)
    assert all((ROOT/p).stat().st_size < 2_000_000 for p in names)
    assets = [r['local_path'] for r in source]
    ignored = subprocess.check_output(['git','check-ignore','--stdin'],input='\n'.join(assets)+'\n',text=True).splitlines()
    assert set(assets) <= set(ignored)
    subprocess.run(['git','diff','--check'],check=True)
    result = dict(protected_files=len(protected['files']),protected_mismatches=bad,
        pinned_source_and_asset_files=len(source),pinned_sources_and_assets_unchanged=True,
        functional_candidates=['EqMotion','SocialVAE'],EqMotion_strict_load_passed=False,
        SocialVAE_strict_load_passed=True,SocialVAE_deployment_causality_passed=False,
        formal_training_completed=False,training_smoke_optimizer_steps=1,ETH_evaluated=False,
        tests=tests,test_failures=failures,paired_target_windows=len(targets),
        original_stages_untouched=True,new_artifacts_below_2MB=True,external_assets_git_ignored=True,
        git_diff_check=True,automatic_push=False,
        git_HEAD_at_check=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if args.record:
        new_json(OUT/'delivery_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
