# Stage11C-0 execution and verification

This stage is a frozen-model semantic intervention, not a training method. The
only source model is the MIT Y-Net commit recorded in the protocol; inherited
HTraj sampling and decoding adapters are read-only. No third-party source is
modified. The generation entry accepts physically GT-free history, frozen
semantics and prediction/reference tensors. The separate CPU scorer alone
loads real validation labels after all six new groups exist.

The executed order is CPU protection/audit, two training-target engineering
smoke checks, committed protocol freeze, S0 bitwise regression, full-chain
S1/S2/S3, fixed-goal S1/S2/S3, then CPU scoring and figures. A first-target
metadata failure was preserved; its one-keyword logging repair was committed
before retry, with no mathematical or configuration changes. Its complete GPU
process wall time remains in the budget.

Use the inherited isolated Python environment and `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`. GPU work must go through `launch`, which meters the entire
subprocess, not merely kernel time. Four modes exist:

```bash
CUDA_VISIBLE_DEVICES=0 experiments/stage11ap/cache/venv/bin/python -m scripts.stage11c0.launch precheck
CUDA_VISIBLE_DEVICES=0 experiments/stage11ap/cache/venv/bin/python -m scripts.stage11c0.launch regression
CUDA_VISIBLE_DEVICES=0 experiments/stage11ap/cache/venv/bin/python -m scripts.stage11c0.launch full
CUDA_VISIBLE_DEVICES=0 experiments/stage11ap/cache/venv/bin/python -m scripts.stage11c0.launch fixed
```

These are historical execution commands, not permission to rerun the stage or
extend the budget. Completed artifacts use exclusive writes and must not be
overwritten. Reverification can read the existing caches and call pure
functions in `statistics.py`, or run the artifact tests; it needs no new GPU
inference. The protocol contains the exact sampling seeds, masks, valid extents,
floating-point definitions, aggregation and fixed case selection rules.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 experiments/stage11ap/cache/venv/bin/python -m pytest -q scripts/stage11c0/tests
```

Actual full/partial tensor caches and weights remain ignored under
`experiments/`; reports, scalar tables, checksum manifests and figure source
arrays are versioned. `full_head_errors.csv` head numbers identify generated
outputs, not corresponding semantic modes across interventions. Only fixed-goal
comparisons preserve the meaning of each original head's conditioning input.

No next-stage training, reliability gate or other intervention is authorized by
the diagnosis. Read `reports/stage11c0/STAGE11C0_REPORT.md` before proposing it.
