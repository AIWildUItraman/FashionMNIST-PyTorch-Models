import torch.nn.functional as F
from torch import nn
import torch

class Residual(nn.Module):
    def __init__(self, inc, outc, c1, c2, c3, c4, use1x1=False, stride=1):
        super(Residual, self).__init__()
        self.p11 = nn.Conv2d(inc, c1, kernel_size=1, stride=stride)
        self.p21 = nn.Conv2d(inc, c2[0], kernel_size=1, stride=stride)
        self.p22 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) 
        self.p31 = nn.Conv2d(inc, c3[0], kernel_size=1, stride=stride)
        self.p32 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p41 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p42 = nn.Conv2d(inc, c4, kernel_size=1, stride=stride)

        self.con1x1 = nn.Conv2d(inc, outc, kernel_size=1, stride=stride) if use1x1 else None

        self.bn = nn.BatchNorm2d(outc)
    def forward(self, X):
        p1 = F.relu(self.p11(X))
        p2 = F.relu(self.p22(F.relu(self.p21(X))))
        p3 = F.relu(self.p32(F.relu(self.p31(X))))
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
    