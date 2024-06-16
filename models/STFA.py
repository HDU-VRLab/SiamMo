import torch
from torch import nn
from mmengine.registry import MODELS


@MODELS.register_module()
class STFA(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = ConvReluBN(128, 256, 2)
        self.conv2 = ConvReluBN(256, 256, 2)
        self.fuse1 = nn.Sequential(
            ConvReluBN(256, 256),
            ConvReluBN(256, 256),
            ConvReluBN(256, 512, 2),
        )
        self.fuse2 = nn.Sequential(
            ConvReluBN(512, 512),
            ConvReluBN(512, 512),
            ConvReluBN(512, 512, 2),
        )
        self.fuse3 = nn.Sequential(
            ConvReluBN(512, 512),
            ConvReluBN(512, 512),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

    def forward(self, stack_feats):
        feats = [torch.cat(stack_feats.chunk(2, 0), 1)]
        stack_feats = self.conv1(stack_feats)
        feats.append(torch.cat(stack_feats.chunk(2, 0), 1))
        stack_feats = self.conv2(stack_feats)
        feats.append(torch.cat(stack_feats.chunk(2, 0), 1))

        cat_feats = feats[1] + self.fuse1(feats[0])
        cat_feats = feats[2] + self.fuse2(cat_feats)
        cat_feats = self.fuse3(cat_feats)
        return cat_feats


class ConvReluBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False, groups=groups),
            nn.SyncBatchNorm(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )

    def forward(self, x):
        return super().forward(x)
