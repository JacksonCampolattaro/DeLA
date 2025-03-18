import math
from typing import Optional

import torch
from torch import Tensor
from torch_geometric import EdgeIndex
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

import torch_sparse
from torch_geometric.utils import sort_edge_index
from torch_sparse import SparseTensor
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

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return torch_sparse.matmul(adj_t, x, reduce='max')
        # return adj.matmul(x, reduce='max')


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

    edge_index = torch.stack([knn_global.reshape(-1), row.reshape(-1)], dim=0)

    sparse_size = (row.max().item() + 1, row.max().item() + 1) if batch is None else (batch.numel(), batch.numel())
    edge_index, _ = sort_edge_index(edge_index, None, sort_by_row=False)
    return SparseTensor(
        row=edge_index[1],
        col=edge_index[0],
        sparse_sizes=sparse_size,
        is_sorted=True,
        trust_data=True,
    )
    # edge_index = EdgeIndex(edge_index, sparse_size=sparse_size, sorted=True)
    # return edge_index.to_sparse(layout=torch.sparse_csr)


def knn_maxpooling(x, knn):
    B, N, C = x.shape
    _, _, K = knn.shape

    x_flat = x.view(-1, C)
    edge_index = knn_to_edges(knn)
    x_pooled = MaxPoolConvolution(in_channels=C)(x_flat, edge_index)
    return x_pooled.view(B, N, C)


def knn_edge_maxpooling(x, knn):
    return knn_maxpooling(x, knn) - x
