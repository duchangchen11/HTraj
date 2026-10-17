"""Bounded download of the author's public checkpoint archive, no credentials."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import html
import re
import time
import zipfile
import requests
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

CACHE = Path('experiments/stage7p2/cache/pretrained')
OUT = Path('results/tables/stage7p2')
FILE_ID = '1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB'


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    steps = []
    archive = None
    session = requests.Session()
    urls = [f'https://drive.google.com/uc?export=download&id={FILE_ID}',
        f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t']
    params = None
    for i, url in enumerate(urls):
        try:
            response = session.get(url, params=params, stream=True, timeout=(20, 60))
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                text = response.text
                title = re.search(r'<title>(.*?)</title>', text)
                steps.append(dict(endpoint=i+1, status=response.status_code, content_type=content_type,
                    html_title=title.group(1) if title else None, bytes=len(response.content)))
                hidden = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', text))
                if hidden.get('id') == FILE_ID:
                    params = {k: html.unescape(v) for k, v in hidden.items() if k in ['id','export','confirm','uuid']}
                continue
            archive = CACHE/'agentformer_models.zip'
            size = 0
            with archive.open('xb') as f:
                for block in response.iter_content(1024*1024):
                    size += len(block)
                    if size > 800*1024**2:
                        raise ValueError('Archive exceeds fixed 800MiB acquisition limit')
                    f.write(block)
            if not zipfile.is_zipfile(archive):
                raise ValueError('Downloaded response is not a ZIP archive')
            steps.append(dict(endpoint=i+1, status=response.status_code, content_type=content_type, bytes=size,
                archive=str(archive), sha256=sha256(archive)))
            break
        except Exception as error:
            steps.append(dict(endpoint=i+1, error=repr(error)))
            archive = None
    files = []
    if archive:
        target = CACHE/'extracted'
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                relative = Path(item.filename)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe archive path')
                if item.is_dir():
                    continue
                destination = target/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = z.read(item)
                with destination.open('xb') as f:
                    f.write(content)
                files.append(dict(path=str(destination), archive_path=item.filename, bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest()))
    json_new(OUT/'official_checkpoint_acquisition.json', dict(author_public_url=urls[0],
        download_steps=steps, available=bool(archive), files=files,
        Baidu_alternate='Author README provides a public alternative; not used when Drive succeeds; no account credentials requested',
        seconds=time.perf_counter()-started, no_evaluation_performed=True))
    print('Author checkpoint archive available:', bool(archive), 'files', len(files), 'seconds', time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    main()
