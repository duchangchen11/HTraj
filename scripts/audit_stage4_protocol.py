"""Compare non-ETH source arrays and window supports without mixing scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib
import numpy as np
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new


def main():
    root=Path('results/tables/stage4');old=json.loads(Path('configs/stage3p8/protocol.json').read_text())
    mapping=dict(biwi_hotel='hotel',students001='univ',students003='univ',crowds_zara01='zara1',crowds_zara02='zara2')
    rows=[]
    for rec,fold in mapping.items():
        assert rec!='biwi_eth'
        official=Path('data/external/stage4/assets/social_stgcnn/datasets')/fold/'test'/(rec+'.txt')
        info=old['source_files'][rec];own=Path(old['raw_root'])/info['file']
        a=np.loadtxt(official);b=np.loadtxt(own);assert a.shape[1]==b.shape[1]==4
        ia=np.lexsort((a[:,1],a[:,0]));ib=np.lexsort((b[:,1],b[:,0]));same_keys=a.shape==b.shape and np.array_equal(a[ia,:2],b[ib,:2])
        manifest=json.loads(Path('configs/stage3p5/folds/val_'+rec+'.json').read_text());wins=manifest['splits']['validation']
        frames=np.unique(a[:,0]);intervals=np.diff(frames)
        row=dict(recording=rec,official_fold=fold,official_file_sha256=sha256(official),HTraj_raw_file_sha256=sha256(own),
            official_rows=len(a),HTraj_raw_rows=len(b),official_frames=len(frames),official_unique_agents=len(np.unique(a[:,1])),
            official_frame_first=float(frames[0]),official_frame_last=float(frames[-1]),frame_interval_values='|'.join(map(str,np.unique(intervals).tolist())),
            same_recording_row_keys=same_keys,max_coordinate_difference_m=float(np.max(np.abs(a[ia,2:]-b[ib,2:]))) if same_keys else None,
            HTraj_manifest_validation_windows=len(wins),HTraj_validation_first_support=min(w['start_frame'] for w in wins),
            HTraj_validation_last_support=max(w['end_frame'] for w in wins),
            interpretation='matching raw file does not imply matching target/context/window protocol; official test whole recording, HTraj original validation region only')
        rows.append(row)
    csv_new(root/'raw_protocol_comparison.csv',rows)
    # License and network/setup evidence: no inference metrics loaded here.
    attempts=[]
    for name in ('environment_setup.log','source_fetch.log','source_fetch_attempt2.log','source_fetch_attempt3.log','eq_asset_fetch.log','literature_discovery.log'):
        path=Path('experiments/stage4')/name
        if path.exists():attempts.append(dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size))
    write_json_new(root/'setup_provenance.json',dict(logs=attempts,
        source_fetch='initial Git TLS errors; bounded retry/raw HTTPS fallback verified against frozen Git blob SHA1; already retrieved identical snapshots reused, no old source modified',
        metadata_inventory='partial clone ls-tree --long size lookup stopped when it triggered lazy object retrieval; all final inventory uses names only; no ETH trajectory/metrics checked out, parsed or evaluated',
        literature_skill='nature-academic-search verification and targeted search; scholarly MCP unavailable, OpenAlex fallback failed SSL; official CVF/arXiv/author repositories used instead',
        environment='independent offline-cloned conda prefix; no packages or source files in the old environment modified',
        data_license='both source repos MIT code; no separate dataset redistribution grant established by that code license; raw/checkpoints/caches stay ignored/local',ETH_evaluated=False))
    print(json.dumps(rows))


if __name__=='__main__':main()
