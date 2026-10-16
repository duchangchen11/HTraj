"""Final read-only evidence/protection audit, no model execution or training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,hashlib,subprocess,time,xml.etree.ElementTree as ET
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

ROOT=Path('results/tables/stage7')


def main():
    begin=time.perf_counter();protected=json.load(open('configs/stage7/protected_artifacts.json'))
    failures=[r['path'] for r in protected['files'] if not Path(r['path']).is_file() or sha256(r['path'])!=r['sha256']]
    assert not failures,failures
    print('Historical SHA256 unchanged:',len(protected['files']),flush=True)
    public=[]
    for name in ['AgentFormer','PECNet']:
        original=json.load(open(ROOT/f'sources/{name}.json'));entries={r['path']:r for r in original['files'] if 'error' not in r}
        if name=='AgentFormer':
            retry=json.load(open(ROOT/'sources/AgentFormer_single_retry.json'));assert retry['success']
            entries.update({r['path']:r for r in retry['attempts']})
        assert len(entries)==len(original['files'])
        for r in entries.values():
            data=Path(r['local_path']).read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256']
            assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==r['git_blob']
            assert subprocess.run(['git','check-ignore','-q',r['local_path']]).returncode==0
        public.append(dict(candidate=name,commit=original['commit'],verified_source_files=len(entries)))
    candidate_results=[];random_checkpoints=[]
    for file in ['AgentFormer_compat_admission.json','PECNet_native_admission.json']:
        result=json.load(open(ROOT/file));assert result['all_cases_passed']
        assert result['formal_training_runs']==result['optimizer_updates']==0 and result['ETH_evaluated'] is False
        assert not result['pretrained_checkpoint_tested']
        assert result['source_SHA256']==sha256('scripts/stage7/admit.py')
        assert [r['people'] for r in result['cases']]==[1,2,5,12]
        for r in result['cases']:
            assert r['passed'] and r['gradients_finite'] and r['strict_random_checkpoint_restore']
            assert r['poisoned_future_max_difference']==r['restored_prediction_max_difference']==0
            assert sha256(r['checkpoint'])==r['checkpoint_sha256']
            random_checkpoints.append(r['checkpoint'])
        candidate_results.append(dict(candidate=result['candidate'],parameter_count=result['parameter_count'],passed=True))
    full=json.load(open(ROOT/'AgentFormer_full_prior_checks.json'))
    assert full['all_passed'] and full['optimizer_updates']==0 and not full['ETH_evaluated']
    assert full['source_SHA']==sha256('scripts/stage7/check_full_prior.py')
    for r in full['cases']:
        assert r['passed'] and r['future_poison_max']==r['strict_restore_max']==0
        assert sha256(r['checkpoint'])==r['checkpoint_sha256'];random_checkpoints.append(r['checkpoint'])
    resource=json.load(open(ROOT/'AgentFormer_max_context_resource_probe.json'))
    assert resource['passed'] and resource['finite_loss'] and resource['gradients_finite'] and resource['people']==57
    assert resource['optimizer_updates']==resource['formal_training_runs']==0 and not resource['ETH_evaluated']
    context=json.load(open(ROOT/'official_context_selection_test.json'))
    assert context['history_equal'] and not context['strict_history_only_context_selection']
    assert sha256(context['source_path'])==context['source_sha256']
    native=json.load(open(ROOT/'AgentFormer_native_admission.json'))
    assert not native['all_cases_passed'] and 'ImportError' in native['setup_traceback']
    first=ET.parse(ROOT/'first_full_tests.xml').getroot().findall('testsuite')
    assert sum(int(r.attrib['failures']) for r in first)==4
    suites=ET.parse(ROOT/'full_tests.xml').getroot().findall('testsuite')
    tests={k:sum(int(r.attrib.get(k,0)) for r in suites) for k in ['tests','errors','failures','skipped']}
    assert tests==dict(tests=385,errors=0,failures=0,skipped=0)
    plan=json.load(open('configs/stage7/minimal_reproduction_plan.json'))
    assert not plan['training_started'] and plan['held_out_ETH'] and plan['max_runs']==4
    assert not plan['independent_confirmation'] and not plan['ETH_evaluated']
    for rec,entry in plan['manifests'].items():
        assert sha256(entry['path'])==entry['sha256']
        m=json.load(open(entry['path']));assert m['future_existence_conditions_context'] and not m['strict_history_context_protocol']
        train={r['recording'] for r in m['splits']['train']};val={r['recording'] for r in m['splits']['validation']}
        assert rec not in train and val=={rec}
        assert not (train|val)-{'biwi_hotel','crowds_zara01','crowds_zara02','students001','students003'}
        assert any(len(r['agent_ids'])==1 for r in m['splits']['validation'])
        for source in m['source_files'].values():assert sha256(Path('data/raw/eth_ucy')/source['file'])==source['sha256']
    decision=json.load(open(ROOT/'final_decision.json'))
    assert decision['new_paper_algorithm_development']=='NO-GO'
    assert decision['component_technical_admission'] and not decision['strict_history_context_pipeline_admitted']
    assert not decision['official_pretrained_strictly_verified'] and not decision['formal_training_started']
    assert decision['frozen_future_plan_sha256']==sha256('configs/stage7/minimal_reproduction_plan.json')
    report_names=['01_HISTORICAL_RESEARCH_CONCLUSIONS.md','02_PUBLIC_BASELINE_COMPARISON.md',
        '03_BASELINE_TECHNICAL_ADMISSION.md','04_PAPER_INNOVATION_FEASIBILITY.md','05_FINAL_GO_NO_GO.md','STAGE7_REPORT.md']
    assert all((Path('reports/stage7')/p).stat().st_size>300 for p in report_names)
    historical={r['path'] for r in protected['files']}
    assert not historical & set(subprocess.check_output(['git','diff','HEAD','--name-only'],text=True).splitlines())
    subprocess.run(['git','diff','--check'],check=True)
    json_new(ROOT/'delivery_verification.json',dict(base_commit=protected['base_commit'],historical_SHA_verified=len(protected['files']),
        historical_SHA_failures=failures,source_repositories=public,candidates=candidate_results,
        random_checkpoint_files_verified=len(random_checkpoints),pretrained_weights_verified=False,
        native_import_failure_preserved=True,first_four_index_test_failures_preserved=True,final_full_tests=tests,
        strict_causal_context_admitted=False,formal_training_runs=0,optimizer_updates_on_candidates=0,ETH_evaluated=False,
        old_research_routes_frozen=True,algorithm_development='NO-GO',future_reproduction_plan_only=True,
        source_SHA={str(p):sha256(p) for p in list(Path('scripts/stage7').glob('*.py'))+list(Path('src/public_admission').glob('*.py'))+[Path('tests/test_public_admission.py')]},
        inline_probes='N57 and official loader ID filter were first executed inline; equivalent bodies persisted as resource_probe.py/protocol_selection_probe.py for reproduction; no second formal run',
        automatic_push=False,elapsed_seconds=time.perf_counter()-begin,
        final_commit='Containing Git delivery commit; use recorded source SHA without circular self-reference'))
    print('All delivery checks passed:',tests,'history',len(protected['files']),flush=True)


if __name__=='__main__':main()
