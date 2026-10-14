"""Audit actual Stage3 protocol, artifacts, tests and preservation (stdlib)."""
import csv,hashlib,json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path('results/tables/stage3')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protected=json.loads(Path('configs/stage3/protected_prior_artifacts.json').read_text())
    for row in protected['files']:
        assert Path(row['path']).is_file(),row['path']
        assert digest(row['path'])==row['sha256'],row['path']
    p=json.loads(Path('configs/stage3/protocol.json').read_text());assert p['epochs']==30 and not p['test_evaluation'] and p['max_order']==1
    selection=json.loads((ROOT/'round2_selection.json').read_text());chosen=selection['selected_models'];assert len(chosen)<=2
    expected={(name,17) for name in p['new_models']}|{(name,seed) for name in chosen for seed in (42,2026)}
    actual={(path.parent.parent.name,int(path.parent.name.split('_')[1])) for path in (ROOT/'new').glob('*/seed_*/summary.json')};assert actual==expected,(actual,expected)
    total_epochs=0
    for name,seed in sorted(actual):
        run=ROOT/'new'/name/f'seed_{seed}';s=json.loads((run/'summary.json').read_text());c=json.loads((run/'config.json').read_text());rows=list(csv.DictReader((run/'epochs.csv').open()))
        assert len(rows)==30 and [int(r['epoch']) for r in rows]==list(range(1,31))
        assert s['optimizer_steps']==3840 and not s['test_evaluated'];total_epochs+=len(rows)
        assert digest(p['data_manifest'])==c['data_manifest_sha256'] and digest(p['subset_manifest'])==c['subset_manifest_sha256']
        assert digest('configs/stage3/protocol.json')==c['protocol_sha256']
        for path,sha in c['source_sha256'].items():
            assert digest(path)==sha,path
            assert hashlib.sha256(subprocess.check_output(['git','show',c['source_commit']+':'+path])).hexdigest()==sha,path
        best=min(rows,key=lambda r:float(r['validation_ADE_m']));assert int(best['epoch'])==s['best_epoch']
        for label in ('best','last'):
            row=best if label=='best' else rows[-1]
            assert abs(float(row['validation_ADE_m'])-s[label+'_ADE_m'])<1e-12
            assert abs(float(row['validation_FDE_m'])-s[label+'_FDE_m'])<1e-12
            checkpoint=Path('experiments/stage3/new')/name/f'seed_{seed}/checkpoints/{label}.pt'
            assert digest(checkpoint)==s[label+'_checkpoint_sha256']
            assert subprocess.run(['git','check-ignore','-q',str(checkpoint)]).returncode==0
            groups={r['group']:r for r in csv.DictReader((run/f'groups_{label}.csv').open())};assert int(groups['all']['ade_count'])==1202 and int(groups['all']['fde_count'])==839
            assert abs(float(groups['all']['ADE_m'])-s[label+'_ADE_m'])<1e-6
            assert abs(float(groups['all']['FDE_m'])-s[label+'_FDE_m'])<1e-6
            assert sum(int(groups[g]['ade_count']) for g in ('history_full8','history_1','history_2to4','history_5to7'))==1202
        d=json.loads((run/'diagnostics_best.json').read_text());assert d['social_messages']>0 and d['unknown_receiver_messages']>0
    conf=json.loads(Path('configs/stage3/confirmation_manifest.json').read_text());assert not conf['enough'] and conf['eligible_windows']==0 and not conf['test_evaluated']
    m=json.loads(Path(p['data_manifest']).read_text());dev=json.loads(Path(p['subset_manifest']).read_text())['validation']['indices'];v=m['splits']['validation']
    assert all(any(w['recording']==v[i]['recording'] and w['start_frame']<=v[i]['end_frame'] and w['end_frame']>=v[i]['start_frame'] for i in dev) for w in v)
    for file in ('pretraining_tests.xml','unit_tests.xml'):
        suites=ET.parse(ROOT/file).getroot();assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors'))
        assert sum(int(s.get('tests',0)) for s in suites)>=140
    for file in ('01_PROBLEM_ANALYSIS.md','02_RECEIVER_FIX_DESIGN.md','03_RELATION_ROUTING_DESIGN.md','04_EXPERIMENT_RESULTS.md','05_ABLATION_ANALYSIS.md','06_RESEARCH_DECISION.md','STAGE3_REPORT.md'):
        assert (Path('reports/stage3')/file).stat().st_size>500
    forbidden=[]
    for path in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],text=True).splitlines():
        f=Path(path)
        if not f.is_file():continue
        if f.suffix in ('.pt','.ckpt','.pth','.npz','.npy') or f.stat().st_size>5*2**20:forbidden.append(path)
    assert not forbidden,forbidden
    assert json.loads((ROOT/'frozen_baseline_audit.json').read_text())['status']=='passed'
    negative=list(csv.DictReader((ROOT/'fixed_checkpoint_interventions.csv').open()));assert len(negative)==10 and all(r['independent_training']=='False' for r in negative)
    print(json.dumps(dict(status='passed',protected_artifacts=len(protected['files']),new_runs=len(actual),epochs=total_epochs,
        frozen_reference_runs=6,auxiliary_inference_cases=len(negative),confirmation_windows=0,test_evaluated=False),sort_keys=True))


if __name__=='__main__':main()
