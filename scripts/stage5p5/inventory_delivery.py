"""Archive completed execution logs and seal only lightweight delivery artifacts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess

from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

ROOT = Path('results/tables/stage5p5')
plan = json.load(open('configs/stage5p5/protocol.json'))
completion = json.load(open(ROOT / 'scheduler_completion.json'))
assert completion['all_succeeded'] and completion['launched_runs'] == 8
verification = json.load(open(ROOT / 'delivery_verification.json'))
assert verification['completed_new_runs'] == 8 and verification['unit_test_failures'] == 0
reports = [
    '01_FROZEN_PROTOCOL.md', '02_MULTISEED_TRAINING.md',
    '03_PAIRED_STATISTICS.md', '04_MOTION_AND_INTERACTION.md',
    '05_GENERALIZATION_AUDIT.md', '06_RESEARCH_DECISION.md',
    'STAGE5P5_REPORT.md',
]
assert all((Path('reports/stage5p5') / name).is_file() for name in reports)

for run in plan['formal_runs']:
    directory = ROOT / 'runs' / f"seed{run['seed']}" / run['model'] / run['recording']
    selection = json.load(open(directory / 'selection.json'))
    assert selection['training_completed'] and selection['completed_epochs'] == 100
    source = directory / 'execution.jsonl'
    with (directory / 'execution.txt').open('xb') as stream:
        stream.write(source.read_bytes())
    assert sha256(source) == sha256(directory / 'execution.txt')

code_paths = sorted(
    p for root in ['scripts/stage5p5', 'src/socialvae_stability']
    for p in Path(root).rglob('*.py') if '__pycache__' not in p.parts
)
code_paths += [Path('tests/test_socialvae_stability.py')]
json_new(ROOT / 'final_analysis_source_sha.json', {
    'pretraining_commit': json.load(open(ROOT / 'runs/seed42/Full/biwi_hotel/provenance.json'))['code_commit'],
    'analysis_parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in code_paths],
    'note': 'Final delivery commit is obtained from Git; no circular self-hash or fabricated post-commit hash.',
})
roots = ['configs/stage5p5', 'reports/stage5p5', 'results/tables/stage5p5',
         'scripts/stage5p5', 'src/socialvae_stability']
paths = {p for root in roots for p in Path(root).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts
         and p.suffix in ['.py', '.json', '.xml', '.txt', '.csv', '.gz', '.md']}
paths |= {Path('tests/test_socialvae_stability.py'),
          Path('experiments/stage5p5/.gitignore'),
          Path('experiments/stage5p5/scheduler_stdout.txt')}
output = ROOT / 'delivery_inventory.json'
paths.discard(output)
for p in paths:
    assert p.stat().st_size < 5_000_000, f'Non-lightweight artifact: {p}'
json_new(output, {
    'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(paths)],
    'weights': 'Ignored locally; each run records all selected/periodic checkpoint paths and SHA256.',
    'raw_data': 'Excluded; manifest and original source SHA256 are retained in frozen config/provenance.',
    'execution_logs': 'Immutable completed JSONL logs archived byte-for-byte as tracked .txt files.',
})
print('Delivery sealed:', len(paths), 'lightweight artifacts; all eight run logs archived')
