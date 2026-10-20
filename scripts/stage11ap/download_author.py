"""Only the user-approved author's Drive resource; no login or CAPTCHA bypass."""
from common import *
from html.parser import HTMLParser
import requests, shutil, zipfile
FILE_ID = '1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa'
VIEW = f'https://drive.google.com/file/d/{FILE_ID}/view'
URL = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download'
class Forms(HTMLParser):
    def __init__(self): super().__init__(); self.action=None; self.fields={}
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='form': self.action=d.get('action')
        if tag=='input' and d.get('name'): self.fields[d['name']]=d.get('value','')
def main():
    dest=CACHE/'ynet_additional_files.zip'
    partial=dest.with_suffix('.zip.part')
    resume = '--resume' in sys.argv
    assert not dest.exists(), 'Do not overwrite prior verified download'
    assert resume or not partial.exists(), 'Use explicit resume for interrupted download'
    offset = partial.stat().st_size if resume and partial.exists() else 0
    free=shutil.disk_usage(ROOT).free
    assert free>10<<30, 'Need at least 10GiB free before inspecting resource'
    log=dict(file_id=FILE_ID,view_url=VIEW,download_url=URL,source_readme=f'https://github.com/HarshayuGirase/Human-Path-Prediction/blob/{MODEL_COMMIT}/ynet/README.md',
             authorized=True,advertised_size='949M (not verified bytes)',disk_free_before=free,events=[],success=False,
             resumed_interrupted_bytes=offset)
    start=time.perf_counter(); session=requests.Session()
    try:
        r=session.get(URL,stream=True,timeout=(15,45))
        log['events'].append(dict(status=r.status_code,content_type=r.headers.get('Content-Type')))
        r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type',''):
            content=r.content; (CACHE/'download_confirmation.html').write_bytes(content)
            p=Forms();p.feed(content.decode('utf-8','replace'))
            log['events'][-1].update(response_bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),form_present=bool(p.action))
            assert b'ynet_additional_files.zip' in content, 'Unexpected filename'
            assert p.action=='https://drive.usercontent.google.com/download', 'Unexpected confirmation action'
            assert p.fields.get('id')==FILE_ID, 'Unexpected file ID'
            # Standard public large-file acknowledgement, not an access/identity challenge.
            r=session.get(p.action,params=p.fields,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,45))
            log['events'].append(dict(status=r.status_code,content_type=r.headers.get('Content-Type'),content_length=r.headers.get('Content-Length'),content_disposition=r.headers.get('Content-Disposition')))
            r.raise_for_status()
        assert 'text/html' not in r.headers.get('Content-Type',''), 'HTML response is not ZIP'
        if offset and r.status_code==206:
            assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-'), 'Invalid resume offset'
            mode='ab'
        elif offset:
            # Preserve interrupted bytes; a server ignoring Range requires a separately named retry.
            log['events'].append(dict(range_ignored=True,interrupted_partial_preserved=str(partial.relative_to(ROOT))))
            partial=dest.with_suffix('.zip.retry2.part'); assert not partial.exists()
            offset=0; mode='xb'
        else: mode='xb'
        total=offset; h=hashlib.sha256(); last=offset
        if offset:
            with partial.open('rb') as f:
                for b in iter(lambda:f.read(8<<20),b''): h.update(b)
        with partial.open(mode) as f:
            for b in r.iter_content(1<<20):
                if not b: continue
                if total==0: assert b[:4]==b'PK\x03\x04', 'Missing ZIP signature'
                total+=len(b); assert total<2<<30, 'Beyond approved approximate resource size'
                f.write(b);h.update(b)
                if total-last>=100<<20:
                    print('Downloaded',round(total/(1<<20),1),'MiB',flush=True);last=total
        assert zipfile.is_zipfile(partial), 'Not a ZIP'
        with zipfile.ZipFile(partial) as z:
            bad=z.testzip(); assert bad is None, f'CRC mismatch: {bad}'
            infos=z.infolist()
        partial.rename(dest)
        log.update(success=True,bytes=total,sha256=h.hexdigest(),mime='application/zip (PK signature + zipfile + CRC)',crc_passed=True,
                   members=len(infos),uncompressed_bytes=sum(x.file_size for x in infos),elapsed_seconds=time.perf_counter()-start)
        print('Verified ZIP',total,'bytes',h.hexdigest(),flush=True)
    except Exception as e:
        log.update(error_type=type(e).__name__,error=str(e),elapsed_seconds=time.perf_counter()-start,
                   partial_bytes=partial.stat().st_size if partial.exists() else 0)
        print(type(e).__name__,str(e),flush=True)
    jnew(OUT/'download_manifest.json',log)
    return 0 if log['success'] else 1
if __name__=='__main__': sys.exit(main())
