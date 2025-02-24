import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

from s3dis import S3DIS, s3dis_test_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

from utils.show_data import show_data

sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from config import s3dis_args, dela_args
from torch.cuda.amp import autocast

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    loop = 12

    testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=loop, train=False, test=True), batch_size=1,
                          collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=8, shuffle=False)

    model = DelaSemSeg(dela_args).cuda()

    util.load_state("pretrained/best.pt", model=model)

    model.eval()

    metric = util.Metric(13)
    cum = 0
    cnt = 0

    with torch.no_grad():
        for xyz, feature, indices, nn, y in testdlr:
            show_data(Data(pos=xyz[nn, :], y=y.flatten()))
            # print(xyz.shape)
            # print(feature.shape)
            # print(nn.shape)
            # print([i.shape for i in indices])
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast():
                p = model(xyz, feature, indices)
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                y = y.cuda(non_blocking=True)
                metric.update(cum, y)
                cnt = cum = 0

    metric.print("test: ")
