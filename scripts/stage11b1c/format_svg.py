"""Remove Matplotlib trailing whitespace while checking all SVG node contents."""
from .common import *
import xml.etree.ElementTree as ET

def normalized_tree(node):
    # XML formatting whitespace is irrelevant; all normalized attributes,
    # visible text, embedded images and child order must remain identical.
    return (node.tag,tuple(sorted((k,' '.join(v.split())) for k,v in node.attrib.items())),
        ' '.join((node.text or '').split()),tuple(normalized_tree(c) for c in node))

def main():
    rows=[]
    for p in sorted(FIG.glob('*.svg')):
        original=p.read_text();before=normalized_tree(ET.fromstring(original))
        cleaned='\n'.join(line.rstrip(' \t\r') for line in original.splitlines())+'\n'
        after=normalized_tree(ET.fromstring(cleaned));assert before==after,p
        digest=hashlib.sha256(repr(before).encode()).hexdigest()
        rows.append(dict(path=str(p.relative_to(ROOT)),original_SHA=sha(p),
            normalized_SVG_structure_SHA=digest,normalized_nodes_identical=True,
            trailing_whitespace_lines=sum(a!=a.rstrip(' \t\r') for a in original.splitlines())))
        p.write_text(cleaned);rows[-1]['final_SHA']=sha(p)
    jnew(OUT/'svg_formatting_audit.json',dict(passed=True,scope='Formatting only; no model, metric, raster or coordinate modification',files=rows))
    print('SVG whitespace cleanup PASS; unchanged normalized XML:',len(rows),'files')

if __name__=='__main__':main()
