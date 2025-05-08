import torch
import torch.nn as nn
import torch.nn.functional as F


class LightSelfAttention(nn.Module):
    """轻量级自注意力模块（参数量仅增加0.15M）"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        # 通道压缩
        self.channel_compress = nn.Conv2d(in_channels, in_channels // reduction, 1)

        # 空间注意力生成
        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_channels // reduction, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 残差缩放系数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 通道压缩
        compressed = self.channel_compress(x)  # [B, C//8, H, W]

        # 生成空间注意力图
        spatial_attn = self.conv_attn(compressed)  # [B, 1, H, W]

        # 特征重加权
        return x + self.gamma * (x * spatial_attn)  # 残差连接


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器（在特征维度较高层添加注意力）
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            LightSelfAttention(128),  # 第一个注意力层
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            LightSelfAttention(256),  # 第二个注意力层
            nn.ReLU()
        )

        # 解码器（保持纯CNN结构）
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