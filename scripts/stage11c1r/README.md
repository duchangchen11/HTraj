# Stage11C-1R resource audit

This completed stage is `RESOURCE_PROFILING_ONLY`. Stage11C-1 remains at zero
formal epochs and zero complete validations. Do not invoke a thirty-epoch run.

The execution protocol was committed at
`f259323d3af3aeacd635192deae1e33fe43f133f` before GPU work. One separate worker
performed the sole authorized diagnostic epoch; another performed six model
states on the same fifteen history-only targets. Every worker lifetime is
charged by its parent, including model construction, CPU counterfactuals and
file output. No additional GPU work is authorized by these scripts.

`observer.py` observes the frozen author's Python call/return events. It does
not replace functions or perform tensor operations inside its callback. Normal
runtime and observed iteration counts come from distinct runs. The CPU-resident
path is only a diagnostic and is never supplied to the path decoder.

`analyze.py`, `supplemental.py`, `plot.py` and `report.py` are CPU-only offline
delivery scripts. Their output creation is exclusive where possible, so a
completed audit should not be casually rerun over its artifacts. Source data,
metadata-only corrections and initial plot versions are retained.

The primary full-prediction monitoring pairs all check RNG before and after.
Six additional frozen-Predictor output comparisons check bitwise predictions
only; their RNG was not separately measured. See `measurement_scope_audit.json`.

The diagnostic checkpoint contains the full Adam and RNG state but is marked
`DIAGNOSTIC_ONLY`. Reuse in a formal run requires separate authorization and
strict continuation verification. CPU migration and KMeans replacement have
not passed the required full998-target equivalence gate.
