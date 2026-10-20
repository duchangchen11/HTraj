# Stage11B-1R reproducible Y-Net sampler

Only the independent sampler adapter is new. Frozen MIT Y-Net functions,
parameters, BCE loss and all previous stages remain unchanged. No optimizer or
formal training runs. The model used here is an explicitly untrained, fixed
seed17 engineering state with frozen author segmentation.

The CPU adapter preserves the original-device float32 weights, relative
threshold0.01, replacement=True,10000 proposals,19 original KMeans centers plus
one softargmax goal and20 decoded paths. CPU and CUDA random sequences differ
by design; their categorical probability definition is held fixed.

`sampling_protocol.json` was committed before observation. Distribution tests
use the fixed50 trials/map/device and the preregistered simultaneous bound.
No validation or test prediction errors are computed.

The completed operations were:

```bash
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1r.protect --freeze
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1r.freeze
# Git commit of protocol and initial diagnostics before GPU execution.
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1r.launch diagnose_original
# Independent adapter implemented and committed after original minimal case.
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1r.launch run_checks
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1r.launch fresh_process
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1r.launch distribution
```

Outputs are exclusive-create. Do not rerun over existing evidence; a new run
needs its own authorized output location. `parent_process_*.json` measures whole
workers, including startup. Do not add the overlapping inner timers a second
time. Workers are serial and stop before the1h budget; full CPU test wall is
also conservatively charged in the final accounting.

Complete regression command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
experiments/stage11ap/cache/venv/bin/python -m pytest tests \
scripts/stage11a/tests scripts/stage11ap/tests scripts/stage11b0/tests \
scripts/stage11b1/tests scripts/stage11b1r/tests -q
```

The recorded full regression has560 passing tests; the subsequently added
empty-cluster fallback test passed separately, covering all561 current tests.
Both XML records are retained. Full outputs retain legacy checkpoint warnings.

Production interface:

```python
from scripts.stage11b1r.predictor import Predictor
# Frozen model,original parameters,original templates; no future argument.
predictor = Predictor(net, cfg, templates)
paths, goals, trace = predictor.predict(sample_id, history8_raw_pixels, image=image)
```

Each call uses one target identity. Caching only the frozen semantic output is
supported and verified. The original KMeans global NumPy/CPU random operators
are wrapped in saved/restored local scopes; use serial execution or isolated
processes if external threads use global RNGs. Bitwise tests cover this hardware
and software stack, not arbitrary future environments. SAMPLER_ENGINEERING_GO
does not authorize training or remove the existing development-protocol limits.
