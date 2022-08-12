import os
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from math import floor, ceil
import numpy as np
import sys
import math
from IPython import embed

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class TCA(nn.Module):
    def __init__(self, in_ch):
        super(TCA, self).__init__()
        self.T_att = nn.AdaptiveAvgPool2d((1, None)) #时间维度的注意力
        self.J_att = nn.AdaptiveAvgPool2d((None, 1)) #坐标维度的注意力
        self.conv2d = nn.Conv2d(in_ch * 3, in_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x是原始的特征图
        h = x.shape[2]
        w = x.shape[3]
        out_1 = self.T_att(x)
        out_1 = out_1.expand(-1, -1, h, w)
        out_2 = self.J_att(x)
        out_2 = out_2.expand(-1, -1, h, w)
        out = torch.cat([x, out_1, out_2], dim=1)

        out = self.conv2d(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, nc=1, out_dim=32, flatten_size=1, initial=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.tca = TCA(nChannels[-1])
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(nc, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[-1])
        self.relu = nn.ReLU(inplace=True)

        self.nChannels = nChannels[3]

        self.conv2 = nn.Conv2d(nChannels[3], out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)    
        self.relu2 = nn.ReLU(inplace=True)

        # self.fc1 = nn.Sequential(nn.Linear(out_dim * 4 * 4, 256 * flatten_size), 
        #                          nn.BatchNorm1d(256 * flatten_size), nn.ReLU())
        # self.fc2 = nn.Sequential(nn.Linear(256 * flatten_size, 128 * flatten_size), 
        #                          nn.BatchNorm1d(128 * flatten_size), nn.ReLU())        
        # self.fc3 = nn.Sequential(nn.Linear(256 * flatten_size, 128 * flatten_size), 
        #                          nn.BatchNorm1d(128 * flatten_size), nn.ReLU())  
        # self.fc4 = nn.Linear(128 * flatten_size, num_classes)

        self.fc1 = nn.Linear(out_dim * 8 * 8, 500 * flatten_size)
        self.fc2 = nn.Linear(500 * flatten_size, 100 * flatten_size)
        self.fc3 = nn.Linear(100 * flatten_size, num_classes)

        # self.fc1 = nn.Linear(1792, 500)
        # self.fc2 = nn.Linear(500, 100)
        # self.fc3 = nn.Linear(100, num_classes) 

        # 初始化
        if initial:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

    def forward(self, x, apply_softmax=False):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)  # --> (1, 256, 61, 61)

        out = self.tca(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = out.view(x.shape[0], -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def EARN(**kwargs):
    """
    construct a wide residual network
    """
    model = WideResNet(**kwargs)
    return model

if __name__ == '__main__':
    model = EARN(depth=28, num_classes=5, widen_factor=4, dropRate=0.4, nc=3)
    x = torch.randn(1,3,244,244)
    print(x.shape)
    model.eval()
    y = model(x)

    print(y.shape)