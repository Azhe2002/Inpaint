import torch
import torch.nn as nn
import torch.fft


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入通道改为5 (RGB + Mask + Frequency Attention)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 64, 5, stride=1, padding=2),  # 修改输入通道为5
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # 解码器保持不变
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        前向传播：
        x: [Batch, 5, Height, Width]，包括 RGB 图像、Mask 和频域特征通道
        """
        x = self.encoder(x)  # 编码特征
        x = self.decoder(x)  # 解码重建图像
        return x


def add_frequency_channel(image, mask):
    """
    计算基于 Mask 的频域注意力通道，并与原始输入合并。

    参数:
    - image: 原始 RGB 图像，形状为 [Batch, 3, Height, Width]
    - mask: 二值化掩码，形状为 [Batch, 1, Height, Width]

    返回:
    - 带有频域通道的输入张量，形状为 [Batch, 5, Height, Width]
    """
    # 将图像和掩码合并为 4 通道
    combined = torch.cat([image, mask], dim=1)  # [Batch, 4, H, W]

    # 计算频域注意力通道
    # 使用快速傅里叶变换 (FFT) 计算频域表示
    fft_image = torch.fft.fft2(image, dim=(-2, -1))
    fft_magnitude = torch.abs(fft_image)  # 幅值谱
    mask_attention = torch.mean(fft_magnitude, dim=1, keepdim=True)  # 基于频域的注意力权重

    # 将频域注意力通道添加到输入
    frequency_channel = mask_attention / (torch.max(mask_attention) + 1e-8)  # 归一化到 [0, 1]
    input_with_frequency = torch.cat([combined, frequency_channel], dim=1)  # [Batch, 5, H, W]

    return input_with_frequency