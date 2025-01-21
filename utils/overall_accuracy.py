import torch
import torchmetrics


def intersection_and_union(pred, label, k, ignore_index=None):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert pred.shape == label.shape
    pred = pred.view(-1)
    label = label.view(-1)
    if ignore_index is not None:
        pred[label == ignore_index] = ignore_index
    intersection = pred[pred == label]
    area_intersection = torch.histc(intersection.float(), bins=k, min=0, max=k - 1)
    area_output = torch.histc(pred.float(), bins=k, min=0, max=k - 1)
    area_target = torch.histc(label.float(), bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class OverallAccuracy(torchmetrics.Metric):
    def __init__(self, num_classes: int, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("intersections", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("targets", default=torch.zeros([num_classes]), dist_reduce_fx="sum")

    def update(self, pred, target, **kwargs):
        if pred.shape != target.shape:
            pred = pred.argmax(dim=-1)

        intersection, _, target = intersection_and_union(
            pred, target, self.num_classes, self.ignore_index
        )
        self.intersections += intersection.to(device=self.intersections.device)
        self.targets += target.to(device=self.targets.device)

    def compute(self, average=None) -> torch.Tensor:
        return self.intersections.sum() / self.targets.sum()
