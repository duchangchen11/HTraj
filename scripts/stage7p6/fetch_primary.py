"""Fetch only public primary literature; never execute downloaded source code."""
import concurrent.futures
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'experiments/stage7p6/cache/literature'
PDFS = {
    'transportmer': 'https://arxiv.org/pdf/2410.17785v2',
    'ms_tip': 'https://raw.githubusercontent.com/mlresearch/v235/main/assets/chib24a/chib24a.pdf',
    'gc_vrnn': 'https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Uncovering_the_Missing_Pattern_Unified_Framework_Towards_Trajectory_Imputation_and_CVPR_2023_paper.pdf',
    'posterior_matching': 'https://arxiv.org/pdf/2201.12414',
    'u2diff': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Capellera_Unified_Uncertainty-Aware_Diffusion_for_Multi-Agent_Trajectory_Modeling_CVPR_2025_paper.pdf',
    'socialvae': 'https://arxiv.org/pdf/2203.08207',
}


def fetch(item):
    name, url = item
    path = CACHE / (name + '.pdf')
    row = dict(name=name, url=url)
    try:
        if path.exists():
            raise FileExistsError(str(path))
        request = urllib.request.Request(url, headers={'User-Agent': 'HTraj research primary-source audit'})
        data = urllib.request.urlopen(request, timeout=60).read()
        if not data.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF; no challenge bypass attempted')
        path.write_bytes(data)
        row.update(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        text_path = path.with_suffix('.txt')
        subprocess.run(['pdftotext', '-layout', str(path), str(text_path)], check=True)
        row['text_path'] = str(text_path.relative_to(ROOT))
        row['text_sha256'] = hashlib.sha256(text_path.read_bytes()).hexdigest()
    except Exception as exc:
        row['error'] = str(exc)
    return row


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch, PDFS.items()))
    path = CACHE / 'pdf_fetch.json'
    with path.open('x') as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
