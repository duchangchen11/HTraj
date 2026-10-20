"""Admission record written only after the new integrated GPU tests pass."""
from .common import *
import xml.etree.ElementTree as ET,importlib.metadata as metadata,platform

def main():
    gate=json.loads((OUT/'preflight_results.json').read_text());assert gate['passed']
    data=json.loads((OUT/'data_audit.json').read_text());assert data['passed']
    root=ET.parse(OUT/'preflight_unit_tests.xml').getroot()
    assert all(int(s.get('errors'))==int(s.get('failures'))==0 for s in root.iter('testsuite'))
    # Source/config changes must already be committed, including the full plot
    # and summary implementation. Result logs may still be untracked.
    dirty=subprocess.check_output(['git','diff','HEAD','--name-only','--','scripts/stage11b1c','configs/stage11b1c'],cwd=ROOT,text=True)
    assert not dirty,dirty
    loose=subprocess.check_output(['git','ls-files','--others','--exclude-standard','scripts/stage11b1c','configs/stage11b1c'],cwd=ROOT,text=True)
    assert not loose,loose
    files=[p for p in (ROOT/'scripts/stage11b1c').rglob('*.py') if '__pycache__' not in p.parts]
    files.extend(ROOT/p for p in ['scripts/stage11b0/runtime.py','scripts/stage11b0/training_adapter.py',
        'scripts/stage11b1r/sampler.py','scripts/stage11b1r/predictor.py'])
    rows=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(files)]
    manifests={s:sha(OUT/f'{s}_target_manifest.csv') for s in ['train','validation']}
    previous=parent_seconds();training=453.825087123*30;validation=.5178994235004211*998*1.1*7
    point=previous+training+validation+600
    high=previous+training*1.3+998*.6387296889024584*1.25*7+900
    jnew(OUT/'budget_preestimate.json',dict(reference_training_epoch_seconds=453.825087123,
        reference_K20_target_seconds=.5178994235004211,point_total_seconds=point,planning_high_seconds=high,
        already_used_GPU_seconds=previous,fit_30_epochs_and7_validations=point<28800,
        assumptions='Prior one full epoch; prior fixed20 target K20;10percent validation overhead; high case1.3training/1.25P90 validation+900s reserve',
        hard_seconds=28800,safety_seconds=27000,measured_new_training_epoch=False,measured_new_validation998=False))
    assert point<28800,'Pre-estimate exceeds budget; do not start'
    jnew(OUT/'protocol_frozen.json',dict(protocol='HTraj-YNet-SDD-Development-v1',status='PROTOCOL_DEVELOPMENT_ONLY',
        freeze_commit=commit(),train_config_SHA=sha(CONF/'train.json'),evaluation_config_SHA=sha(CONF/'evaluation.json'),
        budget_config_SHA=sha(CONF/'budget.json'),sampler_protocol_SHA=sha(ROOT/'configs/stage11b1r/sampling_protocol.json'),
        data_manifest_SHA=manifests,source_manifest_SHA=data['source_manifest_SHA'],executable_files=rows,
        new_integration_GPU_gate_passed=True,training_approved=True,validation_epochs=[1,5,10,15,20,25,30],
        code_not_tuned_to_validation_results=True,Python=platform.python_version(),
        packages={p:metadata.version(p) for p in ['torch','numpy','pandas','matplotlib','segmentation-models-pytorch']}))
    cnew(OUT/'frozen_code_config_SHA256.csv',rows+[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(CONF.glob('*.json'))])
    print('Formal training admitted;point hours',round(point/3600,3),'planning high',round(high/3600,3),flush=True)

if __name__=='__main__':main()
