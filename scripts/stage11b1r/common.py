from pathlib import Path
import csv, gzip, hashlib, json, subprocess, time
from scripts.stage11b0.common import sha, jnew, cnew, commit, RES, OFFICIAL
from scripts.stage11b1.common import configure_torch
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/tables/stage11b1r'
CONF = ROOT/'configs/stage11b1r'
CACHE = ROOT/'experiments/stage11b1r/cache'
BASE = '5c5b6cac4ac8839fd0cd7ff31578a89ac44c17b2'

def timed_run(name, operation):
    """Complete GPU-process timer, with a cumulative conservative55-minute stop."""
    import traceback
    start=time.perf_counter()
    previous=sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('parent_process_*.json'))
    if previous>3000:raise TimeoutError('GPU process budget reserve exhausted')
    def guard(estimated_next=60):
        if previous+time.perf_counter()-start+estimated_next>=3300:
            raise TimeoutError('55-minute cumulative safety stop')
    try:
        operation(guard)
    except Exception as e:
        traceback.print_exc()
        jnew(OUT/f'failure_{name}.json',dict(error_type=type(e).__name__,error=str(e)))
        raise
    finally:
        elapsed=time.perf_counter()-start
        jnew(OUT/f'gpu_process_{name}.json',dict(wall_seconds=elapsed,previous_seconds=previous,
            cumulative_seconds=previous+elapsed,hard_limit_seconds=3600,code_commit=commit()))
