from .common import *

def main():
    assert json.loads((OUT/'checkpoint_static_audit.json').read_text())['passed']
    assert json.loads((OUT/'data_audit.json').read_text())['passed']
    assert json.loads((OUT/'history_before.json').read_text())['passed']
    files=list((ROOT/'scripts/stage11b2b').rglob('*.py'))
    files+=[ROOT/p for p in ['scripts/stage11b0/runtime.py','scripts/stage11b0/training_adapter.py','scripts/stage11b1c/engine.py','scripts/stage11b1c/data.py','scripts/stage11b1r/sampler.py','scripts/stage11b1r/predictor.py']]
    rows=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(files) if '__pycache__' not in p.parts]
    jnew(OUT/'protocol_frozen.json',dict(protocol='HTraj-YNet-SDD-Development-v1',status='CONVERGENCE_DIAGNOSTIC_ONLY',base=BASE,
        freeze_commit=commit(),checkpoint_SHA=CK,official_commit=OFFICIAL,executable_files=rows,
        train_config_SHA=sha(CONF/'train.json'),evaluation_config_SHA=sha(CONF/'evaluation.json'),
        analysis_config_SHA=sha(CONF/'analysis.json'),budget_config_SHA=sha(CONF/'budget.json'),
        data_manifest_SHA={s:sha(OUT/f'{s}_target_manifest.csv') for s in ['train','validation']},
        frozen_analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'results/tables/stage11b2a/motion_features.csv',ROOT/'results/tables/stage11b2a/failure_case_manifest.csv']},
        max_epoch=50,validation_epochs=[35,40,45,50],seed=17,no_test=True,no_hyperparameter_change=True))
if __name__=='__main__':main()
