"""Static diff of serialized old library class source; no checkpoint execution."""
from common import *
import pickletools, importlib, inspect, difflib
def main():
    path=CACHE/'author_resources/ynet_additional_files/segmentation_models/SDD_segmentation.pth'
    with path.open('rb') as f:
        for _ in range(4):ops=list(pickletools.genops(f))
    # Legacy torch module persistent IDs embed original class source as strings.
    globals_=[arg for op,arg,pos in ops if op.name=='GLOBAL']
    sources=[arg for op,arg,pos in ops if isinstance(arg,str) and arg.lstrip().startswith('class ') and '\n' in arg]
    rows=[];diff=[]
    for raw in sources:
        name=raw.lstrip()[6:].split('(')[0].split(':')[0].strip()
        matches=[x for x in globals_ if x.split(' ')[1]==name]
        if not matches:continue
        module,cls=matches[0].split(' ')
        current=inspect.getsource(getattr(importlib.import_module(module),cls))
        changed=raw!=current
        rows.append(dict(class_name=module+'.'+cls,stored_source_SHA=hashlib.sha256(raw.encode()).hexdigest(),current_source_SHA=hashlib.sha256(current.encode()).hexdigest(),source_changed=changed))
        if changed:
            diff.extend(difflib.unified_diff(raw.splitlines(True),current.splitlines(True),fromfile='author_checkpoint/'+module+'.'+cls,tofile='current_runtime/'+module+'.'+cls))
    cnew(OUT/'legacy_class_source_differences.csv',rows)
    with (OUT/'legacy_library_source_diff.txt').open('x') as f:
        f.write('# Display-only unified diff; trailing whitespace normalized. Source SHA records are unchanged.\n')
        f.writelines(line.rstrip()+'\n' for line in diff)
    jnew(OUT/'checkpoint_serialization_audit.json',dict(path=str(path.relative_to(ROOT)),sha256=sha(path),global_names=globals_,
        stored_class_sources=len(sources),compared=len(rows),changed=sum(r['source_changed'] for r in rows),
        sources_executed=False,numerical_equivalence_to_original_runtime='UNVERIFIED',frozen_YNet_model_source_modified=False,
        load_policy='weights_only=True first; only verified official legacy module uses explicit restricted find_class fallback; predictor state dict weights_only=True and strict=True'))
    print('Serialized/current library source comparisons:',len(rows),'changed:',sum(r['source_changed'] for r in rows))
if __name__=='__main__':main()
