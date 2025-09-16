import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, DWConv, DSConv
from ..modules.block import *
from .attention import *


__all__ = ['C3k2_MutilScaleEdgeInformationSelect','DySample']

######################################## DySample start ########################################

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

######################################## DySample end ########################################

######################################## MutilScaleEdgeInformationSelect start ########################################

# 1.使用 nn.AvgPool2d 对输入特征图进行平滑操作，提取其低频信息。
# 2.将原始输入特征图与平滑后的特征图进行相减，得到增强的边缘信息（高频信息）。
# 3.用卷积操作进一步处理增强的边缘信息。
# 4.将处理后的边缘信息与原始输入特征图相加，以形成增强后的输出。
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class EdgeEnhancer_Lite(nn.Module):
    """ 轻量版边缘增强器（修复通道维度） """

    # def __init__(self, in_dim):
    #     super().__init__()
    #     self.filter = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False)
    #     self.scale = nn.Parameter(torch.tensor(0.1))
    #
    # def forward(self, x):
    #     edge = self.filter(x)
    #     return x + self.scale * edge
    def __init__(self, in_dim):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.mask = nn.Conv2d(in_dim, 1, kernel_size=1)  # 得到一个 soft-mask
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        edge = x - self.pool(x)
        mask = torch.sigmoid(self.mask(edge))  # 控制增强区域
        edge = self.conv(edge) * mask
        return x + self.scale * edge



class MutilScaleEdgeInformationSelect(nn.Module):

    def __init__(self, inc, bins):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                DSConv(inc, inc // len(bins), 3)  # 用 DSConv 代替普通 Conv
            ) for bin in bins
        ])

        self.ees = nn.ModuleList([EdgeEnhancer_Lite(inc // len(bins)) for _ in bins])
        self.local_conv = DSConv(inc, inc, 3)  # 局部特征提取也使用 DSConv
        self.dsm = DualDomainSelectionMechanism(inc * 2)
        self.final_conv = DSConv(inc * 2, inc, 3)  # 统一使用 DSConv

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))


class C3k_MutilScaleEdgeInformationSelect(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MutilScaleEdgeInformationSelect(c_, [3, 6, 9, 12]) for _ in range(n)))


class C3k2_MutilScaleEdgeInformationSelect(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MutilScaleEdgeInformationSelect(self.c, self.c, 2, shortcut,
                                                                   g) if c3k else MutilScaleEdgeInformationSelect(
            self.c, [3, 6, 9, 12]) for _ in range(n))

######################################## MutilScaleEdgeInformationEnhance end ########################################