from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from .preprocessing import Recording, sha256


class TrajectoryDataset(Dataset):
    """Manifest windows with a history-only agent set and separate GT masks."""
    def __init__(self, raw_root, manifest_path, split):
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        self.manifest_sha256 = sha256(manifest_path)
        self.windows = manifest['splits'][split] if 'splits' in manifest else manifest['windows']
        self.observation = manifest['observation_frames']
        self.prediction = manifest['prediction_frames']
        self.recordings = {}
        # Load only sources used by this split (no held-out test file in training).
        for name in sorted({x['recording'] for x in self.windows}):
            info = manifest['source_files'][name]
            path = Path(raw_root)/info['file']
            if sha256(path) != info['sha256']:
                raise ValueError('Raw hash changed: '+str(path))
            self.recordings[name] = Recording.read(path, info['scene'])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        entry = self.windows[index]
        sample = self.recordings[entry['recording']].sample(
            entry['start_frame'], self.observation, self.prediction)
        for key, value in sample.items():
            if hasattr(value, 'dtype'):
                sample[key] = torch.from_numpy(value)
        return sample
