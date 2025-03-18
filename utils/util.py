import torch
from torch.nn import Module
from torch_geometric import Index, EdgeIndex
from torch_geometric.data import HeteroData


def load_state(fn, **args):
    state = torch.load(fn)
    for i in args.keys():
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
    return state


def save_state(fn, **args):
    state = {}
    for i in args.keys():
        item = args[i].state_dict() if hasattr(args[i], "state_dict") else args[i]
        state[i] = item
    torch.save(state, fn)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class Metric():
    def __init__(self, num_classes=13, device=torch.device("cuda")):
        self.n = num_classes
        self.label = torch.arange(num_classes, dtype=torch.int64, device=device).unsqueeze(1)
        self.device = device
        self.reset()

    def reset(self):
        # pred == label == i for i in 0...num_classes
        self.intersection = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # pred == i or label == i
        self.union = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # label == i
        self.count = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        #
        self.acc = 0.
        self.macc = 0.
        self.iou = [0.] * self.n
        self.miou = 0.

    def update(self, pred, label):
        """
        pred:  NxC
        label: N
        """
        # CxN
        pred = pred.max(dim=1)[1].unsqueeze(0) == self.label
        label = label.unsqueeze(0) == self.label
        self.tmp_c = label.sum(dim=1)
        self.count += self.tmp_c  # label.sum(dim=1)
        self.intersection += (pred & label).sum(dim=1)
        self.union += (pred | label).sum(dim=1)

    def calc(self, digits=4):
        acc = self.intersection.sum() / self.count.sum()
        self.acc = round(acc.item(), digits)
        macc = self.intersection / self.count
        macc = macc.mean()
        self.macc = round(macc.item(), digits)
        iou = self.intersection / self.union
        self.iou = [round(i.item(), digits) for i in iou]
        miou = iou.mean()
        self.miou = round(miou.item(), digits)

    def print(self, str="", iou=True, digits=4):
        self.calc(digits)
        if iou:
            print(f"{str} acc: {self.acc} || macc: {self.macc} || miou: {self.miou} || iou: {self.iou}")
        else:
            print(f"{str} acc: {self.acc} || macc: {self.macc}")


def from_dela(xyz, x, indices, pts_list=None, y=None, x1_to_x0=None, x0_y=None, x0_pos=None) -> HeteroData:
    indices = [i.long() for i in indices]
    data = HeteroData()

    # Reconstruct batch tensors for each node type
    node_types = ['x3', 'x2', 'x1', 'x0']
    for i, node_type in enumerate(node_types):
        counts = pts_list[i] if pts_list else None
        if not counts:
            continue  # Skip if no nodes for this type
        batch = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long))
        data[node_type].batch = batch

    # If base data is omitted, it'll be the same as x1
    if x1_to_x0 is None:
        x1_to_x0 = torch.arange(xyz.shape[0], device=indices[0].device)

    data['x0'].x = x[x1_to_x0]
    data['x0'].y = x0_y.long() if x0_y is not None else y[x1_to_x0].long()
    data['x0'].pos = x0_pos

    # Assign pos and x to x0
    data['x1'].pos = xyz
    data['x1'].x = x
    data['x1'].y = y.long()

    # Assign selection indices
    data['x1'].selection_index = Index(indices[8])
    data['x2'].selection_index = Index(indices[6])
    data['x3'].selection_index = Index(indices[4])

    # Select points
    data['x2'].pos = data['x1'].pos[data['x1'].selection_index]
    data['x3'].pos = data['x2'].pos[data['x2'].selection_index]
    data['x4'].pos = data['x3'].pos[data['x3'].selection_index]

    # Reconstruct prolongation edges
    # data['x1', 'to', 'x0'].edge_index = EdgeIndex(torch.stack([
    #     x1_to_x0,
    #     torch.arange(data['x0'].num_nodes, device=indices[0].device)
    # ], dim=0))
    # data['x2', 'to', 'x1'].edge_index = EdgeIndex(torch.stack([
    #     indices[0],
    #     torch.arange(data['x1'].num_nodes, device=indices[0].device)
    # ], dim=0))
    # data['x3', 'to', 'x2'].edge_index = EdgeIndex(torch.stack([
    #     indices[1][data['x1'].selection_index],
    #     torch.arange(data['x2'].num_nodes, device=indices[1].device)
    # ], dim=0))
    # data['x4', 'to', 'x3'].edge_index = EdgeIndex(torch.stack([
    #     indices[2][data['x1'].selection_index][data['x2'].selection_index],
    #     torch.arange(data['x3'].num_nodes, device=indices[2].device)
    # ], dim=0))
    data['x1', 'to', 'x0'].edge_index = EdgeIndex(torch.stack([
        x1_to_x0,
        torch.arange(data['x0'].num_nodes, device=indices[0].device)
    ], dim=0))
    data['x2', 'to', 'x1'].edge_index = EdgeIndex(torch.stack([
        indices[0],
        torch.arange(data['x1'].num_nodes, device=indices[0].device)
    ], dim=0))
    data['x3', 'to', 'x1'].edge_index = EdgeIndex(torch.stack([
        indices[1],
        torch.arange(data['x1'].num_nodes, device=indices[1].device)
    ], dim=0))
    data['x4', 'to', 'x1'].edge_index = EdgeIndex(torch.stack([
        indices[2],
        torch.arange(data['x1'].num_nodes, device=indices[2].device)
    ], dim=0))

    # Reconstruct KNN edges
    data['x1', 'to', 'x1'].edge_index = EdgeIndex(torch.stack([
        indices[9].flatten(),
        torch.arange(data['x1'].num_nodes, device=xyz.device).repeat_interleave(24)
    ], dim=0))
    data['x2', 'to', 'x2'].edge_index = EdgeIndex(torch.stack([
        indices[7].flatten(),
        torch.arange(data['x2'].num_nodes, device=xyz.device).repeat_interleave(24)
    ], dim=0))
    data['x3', 'to', 'x3'].edge_index = EdgeIndex(torch.stack([
        indices[5].flatten(),
        torch.arange(data['x3'].num_nodes, device=xyz.device).repeat_interleave(24)
    ], dim=0))
    data['x4', 'to', 'x4'].edge_index = EdgeIndex(torch.stack([
        indices[3].flatten(),
        torch.arange(data['x4'].num_nodes, device=xyz.device).repeat_interleave(24)
    ], dim=0))

    return data


def to_dela(data: HeteroData) -> dict:
    result = {}

    # Extract x, y, and pos from x0 (previously x1)
    result['xyz'] = data['x0'].pos
    result['x'] = data['x0'].x
    result['y'] = data['x0'].y
    result['prev_knn'] = None

    # Extract selection indices (adjusting scales down by one)
    result['indices'] = [
        # data['x1', 'to', 'x0'].edge_index[0],
        # data['x2', 'to', 'x1'].edge_index[0][data['x1', 'to', 'x0'].edge_index[0]],
        # data['x3', 'to', 'x2'].edge_index[0][data['x2', 'to', 'x1'].edge_index[0][data['x1', 'to', 'x0'].edge_index[0]]],
        data['x1', 'to', 'x0'].edge_index[0],
        data['x2', 'to', 'x0'].edge_index[0],
        data['x3', 'to', 'x0'].edge_index[0],

        data['x3', 'to', 'x3'].edge_index[0].reshape(-1, 24),
        data['x2'].selection_index,
        data['x2', 'to', 'x2'].edge_index[0].reshape(-1, 24),
        data['x1'].selection_index,
        data['x1', 'to', 'x1'].edge_index[0].reshape(-1, 24),
        data['x0'].selection_index,
        data['x0', 'to', 'x0'].edge_index[0].reshape(-1, 24),
    ]

    # Extract pts_list if available
    result['pts_list'] = [
        (data[node_type].batch.bincount().tolist() if hasattr(data[node_type], 'batch') else None)
        for node_type in ['x3', 'x2', 'x1', 'x0']
    ]

    return result


def remove_base_scale(data: HeteroData) -> HeteroData:
    scale_map = {f'x{i + 1}': f'x{i}' for i in range(len(data.node_types))}
    scale_unmap = {v: k for k, v in scale_map.items()}

    # Copy features to the reduced databatch
    reduced_data = HeteroData()
    for scale, store in data.node_items():
        if 'x0' == scale:
            continue
        for key, item in store.items():
            reduced_data[scale_map[scale]][key] = item
    for (source, _, dest), store in data.edge_items():
        if 'x0' == source or 'x0' == dest:
            continue
        for key, item in store.items():
            reduced_data[scale_map[source], 'to', scale_map[dest]][key] = item

    return reduced_data
