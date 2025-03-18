import torch_fpsample
import numpy
import torch


def furthest_point_sample(pos, k, start_index=None, approx: bool = True):
    # fixme: round-trips to the CPU! Disgusting.
    device = pos.device
    if len(pos.shape) == 2:
        num_samples = min(k, pos.size(0))
        sample_func = torch_fpsample.sample
        sampled_indices = sample_func(pos, num_samples, start_idx=start_index)
        sampled_indices = torch.pad(sampled_indices, (0, k - len(sampled_indices)))
        return torch.from_numpy(sampled_indices).to(device=device)
    else:
        return torch.stack([
            furthest_point_sample(p, k, start_index)
            for p in pos
        ], dim=0)
