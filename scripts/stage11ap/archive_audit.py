"""Inspect and selectively extract verified resources; no code execution."""
from common import *
import zipfile, stat, shutil

def validate_members(infos, free_bytes):
    seen=set(); total=0
    for x in infos:
        name=x.filename
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe ZIP path: '+name)
        if name in seen: raise ValueError('Duplicate member: '+name)
        seen.add(name)
        if stat.S_ISLNK(x.external_attr>>16): raise ValueError('Symlink member: '+name)
        if x.flag_bits&1: raise ValueError('Encrypted member: '+name)
        if x.file_size>1<<30: raise ValueError('Unexpected member >1GiB: '+name)
        if x.file_size>1<<20 and x.file_size/max(1,x.compress_size)>1000:
            raise ValueError('Abnormal compression ratio: '+name)
        total+=x.file_size
    if total+5*(1<<30)>free_bytes: raise ValueError('Insufficient extraction margin')
    return total

def main():
    package=CACHE/'ynet_additional_files.zip'
    manifest=json.loads((OUT/'download_manifest.json').read_text())
    assert manifest['success'] and sha(package)==manifest['sha256']
    folder=CACHE/'author_resources'; assert not folder.exists()
    with zipfile.ZipFile(package) as z:
        infos=z.infolist();total=validate_members(infos,shutil.disk_usage(ROOT).free)
        rows=[dict(path=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,CRC=f'{x.CRC:08x}',directory=x.is_dir()) for x in infos]
        cnew(OUT/'archive_members.csv',rows)
        summary=dict(archive_sha256=manifest['sha256'],members=len(infos),uncompressed_bytes=total,path_safety=True,
                     abnormal_compression=False,extraction_policy='Only named config, model, image and trajectory data assets; never execute scripts or unpickle during extraction')
        # The bounded, checked archive is not blindly extractall'ed. Only resource types are copied.
        records=[];skipped=[]
        for x in infos:
            if x.is_dir():continue
            if Path(x.filename).suffix.lower() not in {'.yaml','.yml','.json','.txt','.jpg','.jpeg','.png','.pkl','.pickle','.pt','.pth','.ckpt','.csv'}:
                skipped.append(x.filename);continue
            p=folder/x.filename;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(x) as src,p.open('xb') as dst: shutil.copyfileobj(src,dst,8<<20)
            records.append(dict(path=x.filename,bytes=p.stat().st_size,sha256=sha(p)))
        summary.update(extracted=len(records),skipped=skipped)
        cnew(OUT/'extracted_resource_SHA256.csv',records)
        jnew(OUT/'archive_audit.json',summary)
        print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
