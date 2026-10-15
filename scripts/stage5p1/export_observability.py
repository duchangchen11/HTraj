"""Persist per-window visibility/derivative validity without raw coordinates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import json
import numpy as np
from src.data.preprocessing import Recording,sha256,write_json_new
from src.socialvae_causal.data import history_context
from src.socialvae_causal.io import csv_new


def bits(mask):return sum(int(v)<<t for t,v in enumerate(mask))


def main():
    p=json.load(open('configs/stage5p1/protocol.json'));windows=defaultdict(set);files={}
    for fold in p['development_recordings']:
        m=json.load(open(f'configs/stage5p1/manifests/{fold}/B.json'));files.update(m['source_files'])
        for split in m['splits'].values():
            for w in split:windows[w['recording']].add(w['start_frame'])
    results=[]
    for rec,starts in sorted(windows.items()):
        source=files[rec];file=Path(p['data_root'])/source['file'];assert sha256(file)==source['sha256']
        recording=Recording.read(file,source['scene']);rows=[]
        for start in sorted(starts):
            c=history_context(recording,start)
            for i,actor in enumerate(c.ids):
                row=dict(recording=rec,start_frame=start,actor_id=int(actor),history_points=int(c.position_valid[:,i].sum()),
                    position_valid_bits=bits(c.position_valid[:,i]),velocity_valid_bits=bits(c.velocity_valid[:,i]),
                    acceleration_valid_bits=bits(c.acceleration_valid[:,i]),
                    position_only_initial_available=bool(c.position_valid[0,i]),motion_attention_valid_bits=bits(c.velocity_valid[:,i])&254)
                rows.append(row)
        output=Path('results/tables/stage5p1/observability')/(rec+'.csv.gz');csv_new(output,rows)
        results.append(dict(recording=rec,windows=len(starts),actor_windows=len(rows),path=str(output),sha256=sha256(output)))
    write_json_new('results/tables/stage5p1/observability_index.json',dict(files=results,
        bit_order='bit0=earliest raw history frame, bit7=last observation; 1=valid,0=unknown/absent',
        role='out-of-band validity audit, no added network channel',unknown_motion_packets='all six slots padding at attention steps, not fake measured zero velocity',
        initial_position='frame0 position can initialize social hidden state without a velocity claim',
        ego_boundary_acceleration='state[1,4:6]=0 is explicit boundary prior, acceleration_valid=False; not a measured stationary state'))


if __name__=='__main__':main()
