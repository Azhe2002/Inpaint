import torch
import torch.nn as nn
import torch.fft


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入通道改为6 (RGB + Mask + Magnitude + Phase)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 5, stride=1, padding=2),  # 修改输入通道为6
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
        x: [Batch, 6, Height, Width]，包括 RGB 图像、Mask、幅值谱和相位谱
        """
        x = self.encoder(x)  # 编码特征
        x = self.decoder(x)  # 解码重建图像
        return x

def add_frequency_channels(image, mask):
    """
    计算基于 Mask 的频域幅值和相位通道，并与原始输入合并。

    参数:
    - image: 原始 RGB 图像，形状为 [Batch, 3, Height, Width]
    - mask: 二值化掩码，形状为 [Batch, 1, Height, Width]

    返回:
    - 带有频域幅值和相位通道的输入张量，形状为 [Batch, 6, Height, Width]
    """
    # 检查输入形状
    assert image.shape[1] == 3, f"Expected image to have 3 channels, but got {image.shape[1]}"
    assert mask.shape[1] == 1, f"Expected mask to have 1 channel, but got {mask.shape[1]}"

    # 将图像和掩码合并为 4 通道
    combined = torch.cat([image, mask], dim=1)  # [Batch, 4, H, W]

    # 计算频域特征
    # 使用快速傅里叶变换 (FFT) 计算频域表示
    fft_image = torch.fft.fft2(image, dim=(-2, -1))  # [Batch, 3, H, W]
    fft_magnitude = torch.abs(fft_image)  # 幅值谱 [Batch, 3, H, W]
    fft_phase = torch.angle(fft_image)  # 相位谱 [Batch, 3, H, W]

    # 计算幅值和相位的归一化
    magnitude_channel = fft_magnitude.mean(dim=1, keepdim=True)  # 合并到单通道 [Batch, 1, H, W]
    magnitude_channel = magnitude_channel / (torch.max(magnitude_channel) + 1e-8)  # 归一化幅值到 [0, 1]

    phase_channel = fft_phase.mean(dim=1, keepdim=True)  # 合并到单通道 [Batch, 1, H, W]
    phase_channel = (phase_channel + torch.pi) / (2 * torch.pi)  # 归一化相位到 [0, 1]

    # 合并输入通道
    input_with_frequency = torch.cat([combined, magnitude_channel, phase_channel], dim=1)  # [Batch, 6, H, W]

    # 检查输出形状
    assert input_with_frequency.shape[1] == 6, f"Expected output to have 6 channels, but got {input_with_frequency.shape[1]}"
    return input_with_frequency