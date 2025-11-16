# custom_yolo.py
import torch, torch.nn as nn
from ultralytics.nn.modules import Conv, C3Ghost, SPPF

# ---------- ECA (unchanged) ----------
class ECA(nn.Module):
    def __init__(self, c1, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(c1)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


# ---------- Ghost + ECA ----------
class GhostConvWithECA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, ratio=2, dw_size=3):
        super().__init__()
        c_ = c2 // ratio
        self.primary = Conv(c1, c_, k, s, act='silu')
        self.eca = ECA(c_)
        self.cheap = nn.Conv2d(c_, c2 - c_, dw_size, groups=c_,
                               padding=dw_size // 2, bias=False)

    def forward(self, x):
        y = self.primary(x)
        y = self.eca(y)
        return torch.cat([y, self.cheap(y)], 1)


# ---------- Real backbone ----------
class CottonBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 32, 6, 2, 2, act='silu')

        self.stage1 = nn.Sequential(GhostConvWithECA(32, 64, 3, 2),  C3Ghost(64, 64, n=1))
        self.stage2 = nn.Sequential(GhostConvWithECA(64, 128, 3, 2), C3Ghost(128, 128, n=2))
        self.stage3 = nn.Sequential(GhostConvWithECA(128, 256, 3, 2),C3Ghost(256, 256, n=3))
        self.stage4 = nn.Sequential(GhostConvWithECA(256, 512, 3, 2),C3Ghost(512, 512, n=1), SPPF(512, 512))

    def forward(self, x):
        x = self.stem(x)      # 0
        c2 = self.stage1(x)   # 1  (P/4)
        c3 = self.stage2(c2)  # 2  (P/8)
        c4 = self.stage3(c3)  # 3  (P/16)
        c5 = self.stage4(c4)  # 4  (P/32)
        return [c2,c3, c4, c5]   # ← head needs only 3 tensors
        # indices 2,3,4  (see yaml)


class CottonBackboneWrapper(nn.Module):
    """Ultralytics-compatible: returns 5 tensors + tells parser their channels"""
    def __init__(self):
        super().__init__()
        self.backbone = CottonBackbone()
        # we will return [stem, c2, c3, c4, c5]  → indices 0-4
        self.out_channels = [32, 64, 128, 256, 512]

    def forward(self, x):
        x = self.backbone.stem(x)   # 0
        c2 = self.backbone.stage1(x)   # 1
        c3 = self.backbone.stage2(c2)  # 2
        c4 = self.backbone.stage3(c3)  # 3
        c5 = self.backbone.stage4(c4)  # 4
        return [x, c2, c3, c4, c5]       # 5 items → indices 0-4