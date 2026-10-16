"""Frozen official model, runtime import shim and deterministic noise controls."""
from contextlib import contextmanager
import json
import subprocess
import threading
import time
from unittest.mock import patch
import numpy as np
import torch
from scripts.stage7.admit import agentformer
from src.agentformer_admission.causal import inference_packet, stable_epsilon
from src.socialvae_causal.data import array_sha


def load_model():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    meta, build, _, _, settings = agentformer(True)
    return build().eval(), build, meta, settings


def predict(model, context, k=1):
    from model.common.dist import Normal
    model.eval()
    model.set_data(inference_packet(context))
    epsilon = torch.zeros(1, len(context.ids), 32) if k == 1 else stable_epsilon(context, k)

    def sample(distribution):
        e = epsilon.permute(1, 0, 2).reshape(-1, 32).to(distribution.mu)
        assert e.shape == distribution.mu.shape
        return distribution.mu + distribution.sigma*e

    with torch.no_grad(), patch.object(Normal, 'sample', sample):
        output, _ = model.inference(mode='infer', sample_num=k)
    prediction = output.detach().cpu().numpy()
    assert np.isfinite(prediction).all()
    fingerprints = {key: array_sha(model.data[key].detach().cpu().numpy())
        for key in ['pre_motion', 'pre_vel', 'scene_orig', 'agent_mask', 'context_enc']}
    return prediction, fingerprints


class ResourceMeter:
    """Whole-card memory is polled; allocated/reserved are exact torch peaks."""
    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.values = []
        self.done = threading.Event()
        def poll():
            while not self.done.is_set():
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True)
                if result.returncode == 0:
                    self.values.append(int(result.stdout.strip().splitlines()[0]))
                self.done.wait(.1)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        self.done.set()
        self.thread.join()
        self.result = dict(seconds=self.seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            whole_card_sampled_peak_MiB=max(self.values) if self.values else None,
            whole_card_memory_samples=len(self.values), whole_card_sampling_interval_seconds=.1,
            OOM=bool(exc[0] and issubclass(exc[0], torch.cuda.OutOfMemoryError)))
