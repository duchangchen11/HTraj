"""Record immutable critical numerical code and data before GPU extraction."""
from .common import *
import platform,importlib.metadata as md,xml.etree.ElementTree as ET

def main():
    assert json.loads((OUT/'input_export.json').read_text())['passed']
    assert json.loads((OUT/'history_before.json').read_text())['passed']
    tests=ET.parse(OUT/'preflight_unit_tests.xml');assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in tests.iter('testsuite'))
    files=list((ROOT/'scripts/stage11b2c').rglob('*.py'))+[ROOT/p for p in ['scripts/stage11b1r/sampler.py','scripts/stage11b1r/predictor.py','scripts/stage11b0/runtime.py','scripts/stage11b0/training_adapter.py']]
    rows=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(files) if '__pycache__' not in p.parts]
    jnew(OUT/'protocol_frozen.json',dict(freeze_commit=commit(),base=BASE,analysis_protocol_SHA=sha(CONF/'analysis_protocol.json'),checkpoint_SHA=CK,original_prediction_SHA=PSHA,
        executable_files=rows,input_export_SHA=sha(OUT/'input_export.json'),sampler_protocol_SHA=sha(ROOT/'configs/stage11b1r/sampling_protocol.json'),
        historical_motion_labels_SHA=sha(ROOT/'results/tables/stage11b2a/motion_features.csv'),official_commit=OFFICIAL,
        Python=platform.python_version(),packages={k:md.version(k) for k in ['torch','numpy','pandas','matplotlib','segmentation-models-pytorch']},
        no_training=True,no_GT_in_GPU_inputs=True,all998_double_pass_required=True))
if __name__=='__main__':main()
