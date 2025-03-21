import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from s3dis import S3DIS, s3dis_test_collate_fn, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

from utils.show_data import show_data

sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from config import s3dis_args, dela_args
from torch.cuda.amp import autocast

torch.set_float32_matmul_precision("high")

loop = 1

if __name__ == '__main__':
    testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=loop, train=False, test=True), batch_size=1,
                          collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=0)
    valdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=1, train=False), batch_size=1,
                         collate_fn=s3dis_collate_fn, pin_memory=True,
                         persistent_workers=True, num_workers=16)

    model = DelaSemSeg(dela_args).cuda()

    util.load_state("pretrained/best.pt", model=model)

    model.eval()

    # metric = util.Metric(13)
    # cum = 0
    # cnt = 0
    # with torch.no_grad():
    #     for xyz, feature, indices, pts, y in valdlr:
    #             xyz = xyz.cuda(non_blocking=True)
    #             feature = feature.cuda(non_blocking=True)
    #             indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
    #             with autocast():
    #                 p = model(xyz, feature, indices)
    #             cum = cum + p
    #             cnt += 1
    #             if cnt % loop == 0:
    #                 y = y.cuda(non_blocking=True)
    #                 metric.update(cum, y)
    #                 cnt = cum = 0
    #
    # metric.print("val: ")

    metric = util.Metric(13)
    cum = 0
    cnt = 0
    with torch.no_grad():
        for xyz, feature, indices, nn, full_y, full_xyz, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast():
                p = model(xyz, feature, indices)
                #y_one_hot = torch.nn.functional.one_hot(y.long()).to(device=p.device)
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                # show_data(HeteroData(
                #     x0=dict(
                #         pos=full_xyz * 40,
                #         y=full_y,
                #         x=cum,
                #     ),
                #     x1=dict(
                #         pos=xyz,
                #         color=feature[:, :3],
                #         height=feature[:, 3],
                #         x=p,
                #         y=y,
                #     ),
                #     x1__to__x0=dict(
                #         edge_index=torch.stack([
                #             nn.cpu(),
                #             torch.arange(nn.size(0))
                #         ])
                #     ),
                # ))
                full_y = full_y.cuda(non_blocking=True)
                metric.update(cum, full_y)
                cnt = cum = 0

    metric.print("test: ")
