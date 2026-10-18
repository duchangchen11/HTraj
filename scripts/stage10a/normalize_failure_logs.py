"""Preserve raw failures compressed; normalize only TXT trailing whitespace."""
from common import *


def main():
    rows=[]
    for name in ('initial_structure_failures.txt','second_structure_failures.txt'):
        path=OUT/name
        raw=path.read_bytes()
        archive=OUT/(name+'.gz')
        with archive.open('xb') as f:
            f.write(gzip.compress(raw,mtime=0))
        clean='\n'.join(line.rstrip() for line in raw.decode().splitlines())+'\n'
        path.write_text(clean)
        rows.append(dict(path=name,raw_SHA256=hashlib.sha256(raw).hexdigest(),raw_gzip_path=archive.name,
            raw_gzip_SHA256=sha(archive),normalized_txt_SHA256=sha(path),changes='trailing whitespace only'))
    write_json(OUT/'failure_log_preservation.json',rows)


if __name__=='__main__':
    main()
