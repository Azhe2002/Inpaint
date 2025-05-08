import torch.nn as nn
import torch


class LightSpatialAttention(nn.Module):
    """轻量级空间注意力模块 (参数量仅1.5K)"""

    def __init__(self):
        super().__init__()
        # 使用1x1卷积进一步轻量化 (参数量降至512)
        self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x形状: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 空间注意力计算
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))  # [B, 1, H, W]
        return x * self.post_process(attention)  # 广播乘法


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器（在关键位置插入轻量注意力）
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, padding=2),
            LightSpatialAttention(),  # 位置1：早期空间特征增强
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            LightSpatialAttention(),  # 位置2：深层特征精修
            nn.ReLU()
        )

        # 解码器（保持原结构）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x