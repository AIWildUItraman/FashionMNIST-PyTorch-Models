import torch.nn.functional as F
from torch import nn
import torch

class Residual(nn.Module):
    def __init__(self, inc, outc, c1, c2, c3, c4, use1x1=False, stride=1):
        super(Residual, self).__init__()
        #inception block
        self.p11 = nn.Conv2d(inc, c1, kernel_size=1, stride=stride)
        self.p21 = nn.Conv2d(inc, c2[0], kernel_size=1, stride=stride)
        self.p22 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) 
        self.p31 = nn.Conv2d(inc, c3[0], kernel_size=1, stride=stride)
        self.p32 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.dropout = nn.Dropout2d(0.5)
        self.p41 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p42 = nn.Conv2d(inc, c4, kernel_size=1, stride=stride)
        self.con1x1 = nn.Conv2d(inc, outc, kernel_size=1, stride=stride) if use1x1 else None
        self.bn = nn.BatchNorm2d(outc)
    def forward(self, X):
        p1 = F.relu(self.p11(X))
        p2 = F.relu(self.p22(self.dropout(F.relu(self.p21(X)))))
        p3 = F.relu(self.p32(self.dropout(F.relu(self.p31(X)))))
        p4 = F.relu(self.p42(self.p41(X)))
        Y = self.bn(torch.cat((p1, p2, p3, p4), dim=1))

        if self.con1x1:
            X = self.con1x1(X)
        return F.relu(Y + X) 
def Resnet_block(inc, outc, c1, c2, c3, c4, num_Residuals, first_block=False):
    if first_block:
        assert inc == outc
    blk = []
    for i in range(num_Residuals):
        if i == 0 and not first_block:
            blk.append(Residual(inc, outc, c1, c2, c3, c4, use1x1=True, stride=2))
        else:
            blk.append(Residual(outc, outc, c1, c2, c3, c4))
    return nn.Sequential(*blk)
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size = x.size()[2:])
class FlattenLayer(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
def Resnet_4():
    net = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module('resnet_1', Resnet_block(32, 32, 8, (4, 8), (4, 8), 8, 2, first_block=True))
    net.add_module('resnet_2', Resnet_block(32, 80, 16, (16, 32), (8, 16), 16, 2))
    net.add_module('resnet_3', Resnet_block(80, 192, 32, (32, 64), (32, 64), 32, 2))
    net.add_module('resnet_4', Resnet_block(192, 320, 64, (64, 128), (32, 64), 64, 2))
    net.add_module('global_avg_pool', GlobalAvgPool2d())
    net.add_module('fc', nn.Sequential(FlattenLayer(), nn.Linear(320, 10)))
    return net





