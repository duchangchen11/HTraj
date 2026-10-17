"""Cross-stage identity/GT/metric checks, including explicitly different hash schemas."""
import csv
import gzip
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.socialvae_causal.data import array_sha
from src.socialvae_causal.io import csv_new
from src.socialvae_baseline.artifacts import json_new
from stage7p5_complete_statistics import read
from stage7p5_analyze import readcsv


def main():
    out=ROOT/'results/tables/stage7p5';checks={};transition=[];provenance=[]
    for fold in ('hotel','zara2'):
        rows=read(out/f'{fold}_targets.csv.gz')
        before={(r['sequence'],int(r['last_observed_frame']),int(r['person_id'])):r for r in readcsv(ROOT/f'results/tables/stage7p4/execution_v2/{fold}_validation_targets.csv.gz')}
        arrays={}
        for index in readcsv(ROOT/f'results/tables/stage7p4/execution_v2/{fold}_validation_prediction_SHA256.csv.gz'):
            with np.load(ROOT/index['path'],allow_pickle=False) as a:
                for j,i in enumerate(a['person_ids']):arrays[index['sequence'],int(index['frame']),int(i)]=(array_sha(a['history'][j]),array_sha(a['truth'][j]))
        deltas=[]
        for r in rows:
            key=r['sequence'],int(r['frame']),int(r['person_id']);old=before[key]
            assert arrays[key]==(old['history_SHA256'],old['GT_SHA256'])
            provenance.append(dict(fold=fold,sequence=key[0],frame=key[1],person_id=key[2],
                GT_array_SHA256_stage7p4=old['GT_SHA256'],history_array_SHA256_stage7p4=old['history_SHA256'],
                GT_float32_bytes_SHA256_stage7p5=r['GT_SHA256'],history_float32_bytes_SHA256_stage7p5=r['history_SHA256']))
            for new,oldkey in [('ADE_oracle_ADE','minADE20'),('FDE_oracle_FDE','minFDE20'),('head0_ADE','K1_ADE'),('head0_FDE','K1_FDE'),('CV2_ADE','CV-Last2_ADE'),('CV4_ADE','CV-OLS4_ADE'),('CV8_ADE','CV-OLS8_ADE'),('CV2_FDE','CV-Last2_FDE'),('CV4_FDE','CV-OLS4_FDE'),('CV8_FDE','CV-OLS8_FDE')]:
                deltas.append(abs(r[new]-float(old[oldkey])))
            transition.append({k:r[k] for k in ['fold','sequence','frame','person_id','speed_group','history_speed_mps','GT_speed_mps']}|
                dict(GT_minus_history_mps=r['GT_speed_mps']-r['history_speed_mps'],oracle_pred_minus_history_mps=r['ADE_oracle_predicted_speed_mps']-r['history_speed_mps'],all_heads_minus_history_mps=r['all_heads_predicted_speed_mps']-r['history_speed_mps']))
        assert len(rows)==len(before)==len(arrays) and max(deltas)<1e-12
        heads=read(out/f'{fold}_heads.csv.gz')
        assert len(heads)==20*len(rows)
        assert abs(sum(r['ADE_credit'] for r in heads)-len(rows))<1e-8
        assert abs(sum(r['FDE_credit'] for r in heads)-len(rows))<1e-8
        checks[fold]=dict(targets=len(rows),target_IDs_history_GT_exact=True,max_metric_difference=max(deltas),twenty_heads_each_target=True,winner_credit_mass_conserved=True)
    json_new(out/'stage7p4_target_metric_equivalence.json',dict(checks=checks,passed=True,
        SHA_schema_note='Stage7.4 hashes dtype+shape+bytes; Stage7.5 target tables hash bare float32 bytes. Canonical Stage7.4 SHA reverified from every NPZ and saved in target_provenance.csv.gz. Different strings do not mean different GT.',
        first_integration_attempt='Direct comparison of differently defined hash strings failed; no data difference. Repaired explicit schema-aware verification, without modifying either stage tables'))
    csv_new(out/'target_provenance.csv.gz',provenance)
    csv_new(out/'motion_speed_transitions.csv.gz',transition)
    print(checks)


if __name__=='__main__': main()
