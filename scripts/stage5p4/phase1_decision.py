"""Apply frozen four-cell development gate once, no statistical iid claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_capacity.decision import classify
ROOT=Path('results/tables/stage5p4')
plan=json.load(open('configs/stage5p4/ego_protocol.json'));rule=plan['phase1_gate']
rows=read_csv(ROOT/'analysis_phase1/model_metrics.csv')
def lookup(model,rec,weight):
    return next(r for r in rows if r['model']==model and r['recording']==rec and r['weighting']==weight and r['selection']=='best' and r['metric_family']=='K1-prior')
cells=[]
for rec in plan['manifests']:
    for weight in rule['weightings']:
        f=lookup('Full',rec,weight);e=lookup('EgoCapacity',rec,weight)
        fa,ff,ea,ef=map(float,[f['ADE_m'],f['FDE_m'],e['ADE_m'],e['FDE_m']])
        cells.append(dict(recording=rec,weighting=weight,Full_ADE_m=fa,Full_FDE_m=ff,EgoCapacity_ADE_m=ea,EgoCapacity_FDE_m=ef))
result=classify(cells,rule)
# Structural/isolation tests are an additional mandatory gate, not inferred from metrics.
assert Path(ROOT/'pretrain_full_tests.xml').exists()
import xml.etree.ElementTree as ET
suites=ET.parse(ROOT/'pretrain_full_tests.xml').getroot().findall('testsuite')
assert suites and all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
capacity=json.load(open(ROOT/'capacity_audit.json'))[0]
assert abs(capacity['structural_effective_upper_bound']/2045698-1)<=.05
json_new(ROOT/'phase1_decision.json',dict(frozen_rule=rule,**result,
    threshold_not_statistical_significance=True,additional_seeds=False,ETH_evaluated=False))
print(json.dumps(result,indent=2))
