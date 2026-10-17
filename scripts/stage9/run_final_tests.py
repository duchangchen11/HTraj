"""Run required complete regression suite and charge its whole wall time conservatively."""
from common import *
t0=time.monotonic()
environment=os.environ.copy();environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (OUT/'unit_tests.txt').open('x') as f:
    result=subprocess.run([str(ROOT/'experiments/stage4/venv/bin/python'),'-m','pytest','tests','-q','--junitxml='+str(OUT/'unit_tests.xml')],cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
seconds=time.monotonic()-t0;gpu_ledger('final_full_test_suite_conservative_wall',seconds)
json_new(OUT/'unit_test_execution.json',dict(returncode=result.returncode,wall_seconds=seconds,whole_test_wall_charged=True,note='Existing CUDA metric regression included; full test wall is an upper bound on GPU context lifetime. No research optimizer updates from tests.'))
print('Full suite exit',result.returncode,'seconds',seconds)
raise SystemExit(result.returncode)
