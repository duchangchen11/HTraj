"""Final Stage7.6 evidence, regression, preservation, and staging checks."""
import csv
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
OUT=ROOT/'results/tables/stage7p6'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    base='a00c9a38793135b77635ee4f99862f1f017eee23'
    decision=json.loads((OUT/'decision_gates.json').read_text())
    assert decision['decision']=='NO-GO'
    assert decision['terminate_specific_AgentFormer_innovation_route']
    preservation=json.loads((OUT/'historical_preservation.json').read_text())
    assert preservation['passed'] and not preservation['failures']
    cases=ET.parse(OUT/'unit_tests.xml').findall('.//testcase')
    assert len(cases)==442
    assert not ET.parse(OUT/'unit_tests.xml').findall('.//failure')
    assert not ET.parse(OUT/'unit_tests.xml').findall('.//error')
    with (OUT/'supervised_cohort_summary.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    expected={('biwi_hotel','train'):(2121,27145),('biwi_hotel','validation'):(94,318),
              ('crowds_zara02','train'):(1684,23545),('crowds_zara02','validation'):(172,1092)}
    for row in rows:
        assert (int(row['supervised_windows']),int(row['supervised_target_windows']))==expected[row['fold'],row['split']]
        assert int(row['context_node_windows'])==int(row['supervised_target_windows'])+int(row['incomplete_future_node_windows'])
    cf=json.loads((OUT/'counterfactual_reuse.json').read_text())
    assert len(cf['tests'])==10
    for row in cf['tests']:
        assert row['K1_max_difference']==row['K20_max_difference']==0
    source='experiments/stage7/cache/AgentFormer/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5'
    source_files=['data/preprocessor.py','model/agentformer.py','model/agentformer_lib.py',
                  'model/agentformer_loss.py','model/dlow.py']
    csv_new(OUT/'official_source_SHA256.csv',[
        dict(path=source+'/'+p,sha256=sha(ROOT/source/p)) for p in source_files])
    json_new(OUT/'execution_failures.json',dict(
        research_training_failures=[],new_inference_failures=[],
        report_build_failures=[dict(attempt=1,script='scripts/stage7p6/complete.py',
            error="SyntaxError: unmatched ')' at first source row",resolution='Removed duplicated closing delimiters; compilation and execution passed',
            models_executed=False,results_created_before_failure=False)],
        retrieval_failures='literature_sources.json / additional_retrievals.json: HTTP429/403 and unverified repository paths retained',
        historical_negative_probes='negative_probe_reuse.json; preserved as original failed mask-only tests, not new failures',
        limitations='SocialTrans/Inter-GraphFormer full methods not obtained; no novelty approval'))
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=[]
    for directory in ('configs/stage7p6','scripts/stage7p6','reports/stage7p6','results/tables/stage7p6'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.append(ROOT/'tests/test_causal_context_feasibility.py')
    # Existing Stage7.6 files may be changed; no earlier-stage file may be.
    changed=subprocess.check_output(['git','diff',base,'--name-only'],cwd=ROOT,text=True).splitlines()
    allowed=('configs/stage7p6/','scripts/stage7p6/','reports/stage7p6/','results/tables/stage7p6/','tests/test_causal_context_feasibility.py')
    assert all(p.startswith(allowed) for p in changed)
    reports=list((ROOT/'reports/stage7p6').glob('*.md'))
    assert len(reports)==7 and all(p.stat().st_size>0 for p in reports)
    bad_extensions={'.pt','.pth','.ckpt','.npz','.npy','.pkl','.pdf','.pem','.key'}
    patterns=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[0-9A-Z]{16}']
    for path in paths:
        assert path.suffix not in bad_extensions
        assert path.stat().st_size<5_000_000
        if path.suffix!='.gz':
            content=path.read_bytes()
            # Do not scan this detector's literal regex source as a credential.
            assert not any(re.search(pattern,content) for pattern in patterns)
    ignored=['experiments/stage7p6/cache/literature/transportmer.pdf',
             'experiments/stage7p6/cache/literature/posterior-matching/README.md',
             'data/raw/agentformer_official/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/eth/biwi_eth.txt',
             'experiments/stage4/venv/bin/python']
    for path in ignored:
        assert subprocess.run(['git','check-ignore','-q',path],cwd=ROOT).returncode==0
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    excluded={'delivery_SHA256.csv','delivery_checks.json'}
    csv_new(OUT/'delivery_SHA256.csv',[
        dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(paths) if p.name not in excluded])
    json_new(OUT/'delivery_checks.json',dict(base_commit=base,execution_head_commit=current,
        self_containing_final_commit='Final local commit recorded by git log at delivery; no self-referential SHA',
        decision='NO-GO',reports=7,unit_tests_passed=len(cases),unit_tests_failed=0,
        old_SHA_checks=preservation['checked_files'],old_SHA_unchanged=True,
        source_and_data_SHA_recorded=True,paired_counts_verified=True,
        new_model_design_or_training=False,new_predictive_ADE_FDE=False,test_metrics_recomputed=False,
        regression_scope='Full pre-existing suite includes artificial forwards/backwards/optimizer unit tests; not new research training or data inference',
        no_sensitive_or_large_artifacts=True,ignored_inputs=ignored,diff_check_passed=True,
        protocol_freeze_commit='a27cf62',automatic_push=False))
    print('Delivery checks passed:7 reports,442 tests, historical SHA and safe files')


if __name__=='__main__':
    main()
