import torch


def knn_edge_maxpooling(x, knn):
    B, N, C = x.shape
    _, _, K = knn.shape
    batch = torch.arange(B, device=x.device).view(-1, 1, 1).expand(-1, N, K)
    neighbor_features = x[batch, knn]  # Shape: (B, N, K, C)
    pooled_features, _ = neighbor_features.max(dim=2)  # Shape: (B, N, C)
    return pooled_features