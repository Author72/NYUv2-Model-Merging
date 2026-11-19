import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

TASK_CHANNELS = {
    "segmentation": 13,
    "depth": 1,
    "normal": 3,
}

class ResNet50Encoder(nn.Module):
    """ImageNet ResNet-50 with output stride 16 (dilated layer4)."""
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        net = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = 2048

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        branches = [
            ConvBNReLU(in_ch, out_ch, kernel=1, padding=0),
            ConvBNReLU(in_ch, out_ch, kernel=3, padding=rates[0], dilation=rates[0]),
            ConvBNReLU(in_ch, out_ch, kernel=3, padding=rates[1], dilation=rates[1]),
            ConvBNReLU(in_ch, out_ch, kernel=3, padding=rates[2], dilation=rates[2]),
        ]
        self.branches = nn.ModuleList(branches)
        self.pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            # GroupNorm also works for batch_size=1 after global pooling.
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        pooled = self.pool_conv(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))

class TaskHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.aspp = ASPP(in_ch)
        self.refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_ch, 1),
        )

    def forward(self, x, output_size):
        x = self.refine(self.aspp(x))
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=True)

class NYUModel(nn.Module):
    def __init__(self, tasks, pretrained_encoder=False):
        super().__init__()
        self.tasks = list(tasks)
        self.encoder = ResNet50Encoder(pretrained=pretrained_encoder)
        self.heads = nn.ModuleDict({
            task: TaskHead(2048, TASK_CHANNELS[task]) for task in self.tasks
        })

    def forward(self, image):
        feat = self.encoder(image)
        return {
            task: self.heads[task](feat, image.shape[-2:])
            for task in self.tasks
        }
