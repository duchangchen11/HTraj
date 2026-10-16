"""Freeze a named offline full20 cohort, retaining single-agent windows."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from collections import defaultdict
from src.data.preprocessing import sha256
from src.socialvae_causal.data import CausalDataset
from src.socialvae_baseline.artifacts import json_new

plan=json.load(open('configs/stage7/minimal_reproduction_plan.json'));summaries=[];manifests={}
old=json.load(open('configs/stage5p3/protocol.json'))
for rec,source in old['manifests'].items():
    assert sha256(source['path'])==source['sha256']
    splits={};summary=dict(recording=rec)
    for split in ['train','validation']:
        ds=CausalDataset(old['data_root'],source['path'],split)
        grouped=defaultdict(list)
        for wi,ci in ds.indices:grouped[wi].append(int(ds.contexts[wi].ids[ci]))
        rows=[dict(ds.windows[wi],agent_ids=ids) for wi,ids in sorted(grouped.items())]
        splits[split]=rows
        summary[split]=dict(windows=len(rows),targets=sum(len(r['agent_ids']) for r in rows),
            single_person_windows=sum(len(r['agent_ids'])==1 for r in rows),
            max_context_people=max(len(r['agent_ids']) for r in rows),records=sorted(ds.records))
        assert (rec not in ds.records) if split=='train' else set(ds.records)=={rec}
    path=Path('configs/stage7/minimal_manifests')/(rec+'.json')
    json_new(path,dict(protocol=plan['protocol'],source_manifest=source,source_files=ds.manifest['source_files'],splits=splits,
        dt_seconds=.4,obs_len=8,pred_len=12,coordinate_unit='m',context_rule='Complete20 offline supervised agents; includes single-person windows',
        future_existence_conditions_context=True,future_coordinates_used_for_input_features=False,
        strict_history_context_protocol=False,original_other_validation_promoted_to_train=False,ETH_evaluated=False))
    manifests[rec]=dict(path=str(path),sha256=sha256(path));summaries.append(summary)
plan['manifests']=manifests
plan['raw_sources']='Five original nonETH files with Stage5.1 SHA; no old model/cache'
plan['target_cohort']='Same supervised targets as frozen Stage5.1 B, regrouped into synchronous complete20 offline contexts, including N1. Not the Causal-context protocol.'
plan['context_counts']=summaries
# This current-stage design has not been committed or trained; freeze its final
# concrete manifests before delivery. No prior protocol or result is edited.
Path('configs/stage7/minimal_reproduction_plan.json').write_text(json.dumps(plan,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
json_new('results/tables/stage7/minimal_cohort_audit.json',dict(cohorts=summaries,manifests=manifests,ETH_evaluated=False,
    future_existence_conditioning_declared=True,real_prediction_metrics_computed=False,formal_training_runs=0))
print(summaries,flush=True)
