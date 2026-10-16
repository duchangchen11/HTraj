"""Recover operative history mask counts from exhaustive measured aggregates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.socialvae_causal.io import read_csv,csv_new
root=Path('results/tables/stage5p4');stats=read_csv(root/'context_distribution.csv');rows=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for split in ['train','validation']:
  for context in ['real','synthetic']:
   def one(name):return next(r for r in stats if (r['recording'],r['split'],r['context'],r['statistic'])==(rec,split,context,name))
   count=one('count');width=one('context_width')
   total_valid=float(count['mean'])*int(count['observations']);total_slots=float(width['mean'])*int(width['observations'])*8
   assert abs(total_valid-round(total_valid))<1e-5 and abs(total_slots-round(total_slots))<1e-5
   rows.append(dict(recording=rec,split=split,context=context,target_windows=int(count['target_windows']),
    history_frames_per_target=8,history_valid_social_slots=round(total_valid),history_candidate_slots=round(total_slots),
    history_valid_fraction=total_valid/total_slots if total_slots else None,
    history_no_legal_neighbor_frame_fraction=float(count['fraction_zero']),
    synthetic_future_slots_all_padding=context=='synthetic',
    derivation='from exhaustive history-only context_distribution.csv; includes frame0 position initialization, not batch padding'))
csv_new(root/'context_mask_summary.csv',rows)
print('Historical operative mask counts saved')
