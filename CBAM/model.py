import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        sa_map = self.sigmoid(self.conv(concat))
        return sa_map * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, ratio=reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器带CBAM
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            CBAM(64),  # 第一层后加CBAM

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            CBAM(128),  # 第二层后加CBAM

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            CBAM(256)  # 第三层后加CBAM
        )

        # 解码器在关键层加空间注意力
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            SpatialAttention(),  # 只在空间放大后加空间注意力

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x