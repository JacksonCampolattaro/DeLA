import math
from typing import Optional

import torch
from torch import Tensor
from torch_geometric import EdgeIndex
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_scatter import scatter_max


class MaxPoolConvolution(MessagePassing):
    def __init__(self, in_channels: int):
        super().__init__(aggr='max', decomposed_layers=2 if in_channels > 256 else 1)
        self.in_channels, self.out_channels = in_channels, in_channels

    def forward(self, x: Tensor, edge_index: Adj) -> torch.Tensor:
        return self.propagate(
            edge_index,
            x=x,
        )

    def message(self, x_j: Optional[torch.Tensor]) -> torch.Tensor:
        return x_j


def knn_to_edges(knn, batch=None):
    if len(knn.shape) == 3:
        B, N, K = knn.shape
        batch_offsets = torch.arange(B, device=knn.device).view(-1, 1, 1) * N
        knn_global = (knn + batch_offsets).reshape(B * N, K)
        row = torch.arange(B * N, device=knn.device).view(-1, 1).repeat(1, K)

    elif batch is not None:
        N, K = knn.shape
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        batch_offsets = torch.cat([torch.zeros(1, device=knn.device), counts.cumsum(dim=0)[:-1]])
        batch_offsets = batch_offsets[batch].view(-1, 1)  # Expand offsets per point
        knn_global = knn + batch_offsets
        row = torch.arange(N, device=knn.device).view(-1, 1).repeat(1, K)
    else:
        N, K = knn.shape
        knn_global = knn
        row = torch.arange(N, device=knn.device).view(-1, 1).repeat(1, K)

    edge_index = torch.stack([row.reshape(-1), knn_global.reshape(-1)], dim=0)

    sparse_size = (row.max().item() + 1, row.max().item() + 1) if batch is None else (batch.numel(), batch.numel())
    return EdgeIndex(edge_index, sparse_size=sparse_size)


def knn_maxpooling(x, knn):
    B, N, C = x.shape
    _, _, K = knn.shape

    # torch-scatter
    # batch_offset = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    # flat_knn = knn + batch_offset  # Adjust indices for batch
    # flat_knn = flat_knn.view(-1)  # Shape: (B * N * K)
    # x_flat = x.view(B * N, C)  # Flatten x for batch indexing
    # neighbor_features = x_flat[flat_knn]  # Shape: (B * N * K, C)
    # index = torch.arange(B * N, device=x.device).repeat_interleave(K)  # Shape: (B * N * K)
    # pooled_x, _ = scatter_max(neighbor_features, index, dim=0, dim_size=B * N)
    # return pooled_x.view(B, N, C)

    # naive
    # batch = torch.arange(B, device=x.device).view(-1, 1, 1).expand(-1, N, K)
    # neighbor_features = x[batch, knn]  # Shape: (B, N, K, C)
    # pooled_x, _ = neighbor_features.max(dim=2)  # Shape: (B, N, C)
    # return pooled_x

    # PyG
    x_flat = x.view(-1, C)
    edge_index = knn_to_edges(knn)
    x_pooled = MaxPoolConvolution(in_channels=C)(x_flat, edge_index)
    return x_pooled.view(B, N, C)


def knn_edge_maxpooling(x, knn):
    return knn_maxpooling(x, knn) - x
