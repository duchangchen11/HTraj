"""Copy verified raw TXT only and rebuild all five independent fold manifests."""
import argparse
from pathlib import Path
import shutil
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import SCENE_FILES, Recording, sha256, build_manifests, write_json_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/lrj/Mamba-T/data/raw/eth_ucy')
    parser.add_argument('--destination', default='data/raw/eth_ucy')
    parser.add_argument('--manifests', default='configs/data_manifests/stage2_v1')
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    provenance = json.loads((source/'dataset_manifest.json').read_text())
    expected = {item['file']: item['sha256'] for item in provenance['files']}
    destination.mkdir(parents=True, exist_ok=True)
    for scene, names in SCENE_FILES.items():
        for name in names:
            original = source/name
            Recording.read(original, scene)  # Validate contents before copying.
            if sha256(original) != expected[name]:
                raise ValueError('Provenance hash mismatch: '+name)
            target = destination/name
            if target.exists():
                if sha256(target) != sha256(original):
                    raise FileExistsError('Different existing raw file: '+str(target))
            else:
                shutil.copyfile(original, target)
    audit = build_manifests(destination, Path(args.manifests))
    verification = Path('/tmp/htraj-stage2-source-verification.json')
    if verification.exists():
        write_json_new(Path(args.manifests)/'upstream_verification.json', json.loads(verification.read_text()))
    print(json.dumps({ 'raw_files': len(audit['files']), 'manifest_root': args.manifests,
                       'raw_rows': sum(x['rows'] for x in audit['files']) }, indent=2))


if __name__ == '__main__':
    main()
