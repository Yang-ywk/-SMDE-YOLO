import math
import torch
import torch.nn as nn

from ..modules import Conv, DFL, Proto,  DSConv


from ultralytics.utils.tal import dist2bbox, make_anchors
from .attention import *
# from ultralytics.utils.ops import nmsfree_postprocess

__all__ = [ 'Detect_Efficient','Segment_Efficient']

class Detect_Efficient(nn.Module):
    """YOLOv8 Detect Efficient head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    # def __init__(self, nc=80, ch=()):  # detection layer
    #     super().__init__()
    #     self.nc = nc  # number of classes
    #     self.nl = len(ch)  # number of detection layers
    #     self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    #     self.no = nc + self.reg_max * 4  # number of outputs per anchor
    #     self.stride = torch.zeros(self.nl)  # strides computed during build
    #     self.stem = nn.ModuleList(nn.Sequential(Conv(x, x, 3, g=x // 16), Conv(x, x, 3, g=x // 16)) for x in ch) # two 3x3 Group Conv
    #     # self.stem = nn.ModuleList(nn.Sequential(Partial_conv3(x, 4), Conv(x, x, 1)) for x in ch) # one PConv(CVPR2023), one 1x1 Conv
    #     self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
    #     self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
    #     self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 修改点：将原组卷积替换为两个 DSConv
        self.stem = nn.ModuleList(nn.Sequential(
            DSConv(x, x),
            ECA(x),  # 添加ECA
            DSConv(x, x),
            ECA(x)  # 第二个ECA
             ) for x in ch)

        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a.bias.data[:] = 1.0  # box
            b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment_Efficient(Detect_Efficient):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect_Efficient.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))