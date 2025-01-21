import torch
import torch_geometric


def grid_subsampling(xyz: torch.Tensor, grid_size: float) -> torch.Tensor:
    c = torch_geometric.nn.voxel_grid(xyz, grid_size)
    c, perm = torch_geometric.nn.pool.consecutive.consecutive_cluster(c)
    return perm
