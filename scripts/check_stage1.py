#!/usr/bin/env python3
"""Read-only checks for the Stage 1 delivery; never runs a model."""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'README.md', '.gitignore', 'AGENTS.md',
    'docs/literature/01_HIGHGRAPH_ANALYSIS.md',
    'docs/literature/02_UNIEDGE_ANALYSIS.md',
    'docs/literature/03_RELATED_WORK_COMPARISON.md',
    'docs/literature/04_HOGAT_VERIFICATION.md',
    'docs/literature/SOURCE_REGISTER.md',
    'docs/literature/source_manifest.json',
    'docs/literature/references.bib',
    'docs/method_design/00_SHARED_CONTRACT.md',
    'docs/method_design/01_TOHI_NET_DESIGN_A.md',
    'docs/method_design/02_TOHI_NET_DESIGN_B.md',
    'docs/method_design/03_ARCHITECTURE_COMPARISON.md',
    'docs/method_design/04_EXPERIMENT_PLAN.md',
    'docs/thesis/THESIS_OUTLINE.md',
    'docs/thesis/RESEARCH_PROGRESS.md',
    'reports/stage1/STAGE1_REPORT.md',
    'reports/stage1/REPOSITORY_AUDIT.md',
    'configs/stage1_design.yaml',
]
DIRECTORIES = [
    'configs', 'data/raw', 'data/processed', 'src/data', 'src/models',
    'src/training', 'src/evaluation', 'scripts', 'docs/literature',
    'docs/method_design', 'docs/thesis', 'experiments', 'results/tables',
    'results/figures', 'reports/stage1', 'tests',
]


def main():
    errors = []
    for name in REQUIRED:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append('Missing/empty deliverable: ' + name)
    for name in DIRECTORIES:
        if not (ROOT / name).is_dir():
            errors.append('Missing directory: ' + name)
    for path in ROOT.rglob('*.md'):
        if '.git' in path.parts:
            continue
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            destination = target.split('#', 1)[0]
            if destination and not (path.parent / destination).exists():
                errors.append(f'Broken relative link: {path.relative_to(ROOT)} -> {target}')
        if text.count('$$') % 2 or text.count('```') % 2:
            errors.append('Unbalanced math/code delimiter: ' + str(path.relative_to(ROOT)))
    manifest = ROOT / 'docs/literature/source_manifest.json'
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            for item in data['sources']:
                if not item.get('url') or not item.get('verification_scope'):
                    errors.append('Source lacks evidence scope')
        except (ValueError, KeyError) as exc:
            errors.append('Invalid source manifest: ' + str(exc))

    ignored = [
        'data/raw/dataset.txt', 'data/processed/cache.csv', 'model.ckpt',
        'src/models/model.pth', 'weights/example.bin', 'checkpoints/epoch.bin',
        'experiments/test/logs/train.txt', 'experiments/test/cache/data.bin',
        'events.out.tfevents.test', 'run.log', 'run.jsonl', '.env', '.env.local',
        'secrets/account.txt', 'credentials.json', 'private.pem', 'id_rsa',
        'venv/bin/python', '.venv/bin/python', '.conda/bin/python',
        '__pycache__/example.pyc', 'docs/literature/sources/paper.pdf',
    ]
    allowed = [
        'data/raw/.gitkeep', 'data/processed/.gitkeep', 'src/models/example.py',
        'src/training/train.py', 'src/evaluation/metrics.py', 'configs/example.yaml',
        'results/tables/example.csv', 'results/figures/example.png',
        'docs/literature/example.md', 'reports/stage1/example.md',
    ]
    for expected, paths in [(True, ignored), (False, allowed)]:
        for name in paths:
            proc = subprocess.run(
                ['git', 'check-ignore', '-v', '--no-index', name],
                cwd=ROOT, stdout=subprocess.PIPE, text=True,
            )
            # Verbose output also reports negated rules, which mean included.
            pattern = proc.stdout.split('\t', 1)[0].split(':', 2)[-1] if proc.stdout else ''
            is_ignored = bool(pattern) and not pattern.startswith('!')
            if proc.returncode not in (0, 1):
                errors.append('git check-ignore failed: ' + name)
            elif is_ignored != expected:
                errors.append('Incorrect ignore rule: ' + name)

    # Inspect both indexed and new, nonignored files. No secret contents printed.
    proc = subprocess.run(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT, stdout=subprocess.PIPE, check=True,
    )
    files = set(proc.stdout.decode().split('\0')) - {''}
    sensitive = re.compile(
        r'(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
        r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
        r'AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,})'
    )
    total = 0
    largest = ('', 0)
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            errors.append('Review symlink before publication: ' + name)
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > largest[1]:
            largest = (name, size)
        if size > 5 * 1024 * 1024:
            errors.append('File exceeds 5 MiB review threshold: ' + name)
        if sensitive.search(path.read_text(encoding='utf-8', errors='ignore')):
            errors.append('Possible secret pattern; inspect locally: ' + name)
    proc = subprocess.run(['git', 'diff', '--check'], cwd=ROOT)
    staged = subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT)
    if proc.returncode or staged.returncode:
        errors.append('git diff --check failed')
    if errors:
        for error in errors:
            print('FAIL:', error)
        return 1
    print(f'PASS: {len(REQUIRED)} deliverables, {len(DIRECTORIES)} directories, '
          f'{len(ignored) + len(allowed)} ignore probes; relative links and delimiters valid.')
    print(f'PASS: {len(files)} nonignored files, {total} bytes; '
          f'largest {largest[0]} ({largest[1]} bytes); no matched secret patterns.')
    print('Scope: document/repository checks only; no model tests or training executed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
