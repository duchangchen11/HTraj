import csv
import gzip
import io
from pathlib import Path
from src.data.preprocessing import write_json_new


def csv_new(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    rows = iter(rows)
    first = next(rows,None)
    if first is None and fieldnames is None:raise ValueError('Empty CSV needs explicit schema')
    fields = list(first) if fieldnames is None else fieldnames
    with path.open('xb') as raw:
        binary = gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if path.suffix=='.gz' else raw
        with io.TextIOWrapper(binary,encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
            writer.writeheader()
            if first is not None:writer.writerow(first)
            writer.writerows(rows)


def read_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))
