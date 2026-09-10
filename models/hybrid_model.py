import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import config


# ─────────────────────────────────────────
# CBAM Attention
# channel + spatial attention
# ─────────────────────────────────────────
class CBAMAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        # channel attention
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        # spatial attention
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # channel attention
        avg = self.channel_fc(
            self.channel_avg(x).view(b, c)
        ).view(b, c, 1, 1)
        mx  = self.channel_fc(
            self.channel_max(x).view(b, c)
        ).view(b, c, 1, 1)
        x   = x * torch.sigmoid(avg + mx)

        # spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        s     = torch.cat([avg_s, max_s], dim=1)
        x     = x * torch.sigmoid(self.spatial_conv(s))

        return x


# ─────────────────────────────────────────
# Lesion UNet
# detects MA / HE / EX / NV lesions
# outputs 4-channel lesion map
# ─────────────────────────────────────────
class LesionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # encoder
        self.enc1 = self._block(3,  16)
        self.enc2 = self._block(16, 32)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = self._block(32, 64)

        # decoder
        self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._block(64, 32)
        self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._block(32, 16)

        # output — 4 lesion channels
        self.final = nn.Conv2d(16, 4, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


# ─────────────────────────────────────────
# Hybrid Model — Option B
#
# Input (384x384x3)
#   ├── EfficientNet-B5 backbone  ← upgraded from B4
#   │     └── CBAM Attention
#   │           └── AdaptiveAvgPool → 2048  ← was 1792
#   └── LesionUNet
#         └── AdaptiveAvgPool(16x16) → flatten → 1024
#
# Concat [2048 + 1024] = 3072  ← was 2816
#   └── Fusion: Linear→1024, LayerNorm, GELU, Dropout(0.3)
#         └── OrdinalHead → 5 classes
# ─────────────────────────────────────────
class HybridModel(nn.Module):

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        dropout=0.5        # increased from 0.4
    ):
        super().__init__()

        # ── backbone ────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b5",   # ← upgraded from efficientnet_b4
            pretrained=True,
            num_classes=0,
            global_pool=""
        )
        backbone_channels = 2048  # ← B5 output, was 1792 for B4

        # ── CBAM attention ───────────────────────
        self.cbam = CBAMAttention(backbone_channels)

        # ── global pool ──────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── lesion branch ────────────────────────
        self.lesion_unet = LesionUNet()
        self.lesion_pool = nn.AdaptiveAvgPool2d(16)
        lesion_flat      = 4 * 16 * 16   # 1024

        # ── fusion ───────────────────────────────
        fusion_in = backbone_channels + lesion_flat  # 2048+1024=3072
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),   # increased from 0.2
        )

        # ── ordinal head ─────────────────────────
        self.ordinal_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),       # 0.5
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),   # 0.25
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # backbone + attention
        feat    = self.backbone(x)
        feat    = self.cbam(feat)
        feat    = self.global_pool(feat).flatten(1)  # (B, 2048)

        # lesion branch
        lesions = self.lesion_unet(x)
        lesions = self.lesion_pool(lesions).flatten(1)  # (B, 1024)

        # fuse and classify
        fused   = self.fusion(torch.cat([feat, lesions], dim=1))
        return self.ordinal_head(fused)