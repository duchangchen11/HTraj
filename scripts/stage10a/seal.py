"""Final evidence seal and repository safety checks. Does not commit or push."""
from common import *
import xml.etree.ElementTree as ET


def main():
    changes = subprocess.check_output(['git','diff','--name-only','d8e753f718647db49202da3ac6d43e4b37cfc59b'],cwd=ROOT,text=True).splitlines()
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    allow = ['reports/stage10a/','results/tables/stage10a/','configs/stage10a/','scripts/stage10a/','src/risk_dynamic_graph/','tests/test_stage10a_risk_graph.py']
    assert all(any(p.startswith(prefix) for prefix in allow) for p in changes+untracked)
    histories = json.loads((OUT/'history_after.json').read_text())
    assert histories['passed'] and not histories['failures']
    suite = ET.parse(OUT/'final_unit_tests.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.get('failures')) == 0 and int(suite.get('errors')) == 0
    assert int(suite.get('tests')) == 501
    files = []
    for root in allow:
        p = ROOT/root
        files.extend([q for q in p.rglob('*') if q.is_file() and '__pycache__' not in q.parts] if p.is_dir() else [p])
    for p in files:
        assert p.stat().st_size < 12*1024*1024, str(p)
        assert p.suffix not in ('.pt','.pth','.ckpt','.npz','.npy','.pdf','.pkl','.pickle')
    ignored = ['experiments/stage10a/cache/papers/snapshot.pdf','experiments/stage9/cache/checkpoints/A_epoch200.pt','data/raw/eth_ucy/biwi_eth.txt']
    for p in ignored:
        assert subprocess.run(['git','check-ignore','-q',p],cwd=ROOT).returncode == 0
    write_json(OUT/'delivery_checks.json',dict(
        passed=True,unit_tests=int(suite.get('tests')),failures=0,errors=0,
        historical_files_verified=histories['checked'],history_unchanged=True,
        stage10_only_paths=True,largest_file_bytes=max(p.stat().st_size for p in files),
        no_raw_data_weights_or_PDF_staged=True,ignored_samples=ignored,
        formal_training=False,optimizer_updates=0,new_validation_ADE_FDE=False,test_evaluation=False,
        config_frozen_commit='61ee113cdfd9110c644a72d46ed261d2985168b4',
        sealing_execution_commit=commit(),
        log_whitespace='Raw failed pytest output compressed and hashed; txt trailing spaces normalized only',
        research_decision='NO-GO for innovative Stage10B; engineering admission passed',
        push_authorization='Stage10A explicitly requests push new branch, superseding earlier local-only preference'))
    source = [p for p in files if any(str(p.relative_to(ROOT)).startswith(k) for k in [
        'src/risk_dynamic_graph/','scripts/stage10a/','configs/stage10a/','tests/test_stage10a_risk_graph.py'])]
    write_csv(OUT/'final_code_config_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(source)])
    files.extend([OUT/'delivery_checks.json',OUT/'final_code_config_SHA256.csv'])
    # This index excludes itself. No hashing of a still-open stdout stream.
    write_csv(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))])
    print('Stage10A delivery sealed; tests501, historic SHA',histories['checked'])


if __name__=='__main__':
    main()
