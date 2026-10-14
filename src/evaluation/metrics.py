"""K=1 deterministic metrics, averaged per target track; final-horizon FDE."""
import torch


def metric_sums(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError('Prediction/target/mask shape mismatch')
    difference = torch.where(valid[..., None], prediction-target, torch.zeros_like(prediction))
    error = torch.linalg.vector_norm(difference, dim=-1)
    per_track_count = valid.sum(1)
    track_valid = per_track_count > 0
    ade_per_track = error.sum(1)/per_track_count.clamp_min(1)
    final_valid = valid[:, -1]
    return dict(ade_sum=ade_per_track[track_valid].sum(), ade_count=track_valid.sum(),
                fde_sum=error[:, -1][final_valid].sum(), fde_count=final_valid.sum())


def masked_l2_loss(prediction, target, valid):
    values = metric_sums(prediction, target, valid)
    return values['ade_sum']/values['ade_count'].clamp_min(1)
