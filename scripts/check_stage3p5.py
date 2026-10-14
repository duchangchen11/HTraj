"""Stage3.5 preservation, protocol, result and secrecy delivery audit."""
import csv,json,hashlib,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path('results/tables/stage3p5');conf=Path('configs/stage3p5');protected=json.loads((conf/'protected_artifacts.json').read_text())
    for r in protected['files']:assert digest(r['path'])==r['sha256'],r['path']
    old=json.loads(Path('configs/data_manifests/stage2_v1/fold_ETH.json').read_text());oldtrain={w['sample_id'] for w in old['splits']['train']};oldval={w['sample_id'] for w in old['splits']['validation']}
    for file in (conf/'folds').glob('*.json'):
        m=json.loads(file.read_text());train=m['splits']['train'];val=m['splits']['validation'];assert not {w['recording'] for w in train}&{w['recording'] for w in val}
        assert {w['sample_id'] for w in train}<=oldtrain and not {w['sample_id'] for w in train}&oldval
        assert {w['sample_id'] for w in val}<=oldval and all(w['scene']!='ETH' for w in train+val)
        assert all(w['end_frame']-w['start_frame']==190 for w in train+val)
    summaries=list((root/'pilot').glob('*/PedBase/summary.json'))+list((root/'pilot').glob('*/Dynamic-GNN-RN/summary.json'));assert len(summaries)==4
    for path in summaries:
        s=json.loads(path.read_text());c=json.loads(path.with_name('config.json').read_text());rows=list(csv.DictReader(path.with_name('epochs.csv').open()))
        assert len(rows)==30 and int(s['seed'])==17 and not s['ETH_evaluated']
        best=min(rows,key=lambda r:float(r['validation_ADE_m']));assert int(best['epoch'])==s['best_epoch'];assert float(best['validation_FDE_m'])==s['best_FDE_m']
        assert s['protocol_sha256']==digest(conf/'protocol.json')
        for file,sha in c['source_sha256'].items():assert digest(file)==sha,file
        for endpoint in ('best','last'):
            checkpoint=Path('experiments/stage3p5/pilot')/s['protocol_name']/s['model']/f'checkpoints/{endpoint}.pt'
            assert digest(checkpoint)==s[endpoint+'_checkpoint_sha256'];assert subprocess.run(['git','check-ignore','-q',str(checkpoint)]).returncode==0
            groups={r['group']:r for r in csv.DictReader(path.with_name(f'groups_{endpoint}.csv').open())}
            assert int(groups['all']['ADE_count'])==(706 if s['protocol_name'].startswith('A_') else 1464)
            assert int(groups['all']['FDE_count'])==(706 if s['protocol_name'].startswith('A_') else 986)
            assert abs(float(groups['all']['ADE_m'])-s[endpoint+'_ADE_m'])<1e-6
    for name in ('pretraining_tests.xml','unit_tests.xml'):
        suites=ET.parse(root/name).getroot();assert sum(int(s.get('tests',0)) for s in suites)>=148
        assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors'))
    assert json.loads((root/'B_legacy_1024_96_frozen_analysis/compatibility_audit.json').read_text())['status']=='passed'
    assert not json.loads((root/'external_reference_sources.json').read_text())['replicated']
    official=json.loads((root/'external_official_split_inventory.json').read_text())
    assert not official['trajectory_blobs_read'] and not official['ETH_trajectory_or_metrics_read']
    assert official['additional_official_ETH_fold_train_val_recordings']==['crowds_zara03','uni_examples']
    physical=list(csv.DictReader((root/'physical_density_group_comparison.csv').open()))
    assert len(physical)==16
    for setting in {r['setting'] for r in physical}:
        for endpoint in ('best','last'):
            rows=[r for r in physical if r['setting']==setting and r['endpoint']==endpoint]
            assert sum(int(r['ADE_count']) for r in rows)==(706 if setting.startswith('A_') else 1464)
            assert sum(int(r['FDE_count']) for r in rows)==(706 if setting.startswith('A_') else 986)
    assert len(list(csv.DictReader((root/'physical_density_paired_targets.csv').open())))==4340
    for file in ('01_PROTOCOL_AUDIT.md','02_BASELINE_COMPARISON.md','03_INTERACTION_VALUE_ANALYSIS.md','04_TRAINING_FEASIBILITY.md','05_CHAPTER4_RESEARCH_DECISION.md','STAGE3P5_REPORT.md'):assert (Path('reports/stage3p5')/file).stat().st_size>500
    for name in ('legacy','pilot'):
        rows=list(csv.DictReader((root/(name+'_paired_target_differences.csv')).open()));assert len(rows)==(3606 if name=='legacy' else 4340)
    print(json.dumps(dict(status='passed',protected_artifacts=len(protected['files']),fresh_pilot_runs=4,epochs=120,new_training_recordings_disjoint=True,ETH_evaluated=False,external_model_replicated=False)))


if __name__=='__main__':main()
