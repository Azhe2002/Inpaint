import torch.nn as nn
import torch.nn.functional as F


class LightChannelAttention(nn.Module):
    """极简通道注意力模块（参数量<0.1M）"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        att = (avg_out + max_out).view(b, c, 1, 1)
        return x * att.expand_as(x)


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器（仅在关键层后添加注意力）
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, padding=2),
            LightChannelAttention(64),  # 第一层后添加
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            LightChannelAttention(256),  # 最深层次添加
            nn.ReLU()
        )

        # 解码器（保持原样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.decoder(x)
        return x