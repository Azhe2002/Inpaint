import torch
import torch.nn as nn
import torch.fft


class InpaintLoss(nn.Module):
    def __init__(self, use_frequency_loss=True, lambda_freq=0.1):
        """
        初始化损失函数。
        参数:
        - use_frequency_loss: 是否启用频域损失。
        - lambda_freq: 频域损失的权重系数。
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.use_frequency_loss = use_frequency_loss
        self.lambda_freq = lambda_freq

    def forward(self, output, target, mask):
        """
        前向传播计算损失。
        参数:
        - output: 模型输出的修复图像，形状为 [Batch, 3, Height, Width]。
        - target: 目标完整图像，形状为 [Batch, 3, Height, Width]。
        - mask: 掩码，形状为 [Batch, 1, Height, Width]，0 表示破损区域。

        返回:
        - 总损失值。
        """
        # 空间域 L1 损失 (仅考虑破损区域)
        spatial_loss = self.l1_loss(output * (1 - mask), target * (1 - mask))

        # 如果启用频域损失，则计算
        if self.use_frequency_loss:
            freq_loss = self.compute_frequency_loss(output, target)
            total_loss = spatial_loss + self.lambda_freq * freq_loss
        else:
            total_loss = spatial_loss

        return total_loss

    def compute_frequency_loss(self, output, target):
        """
        计算频域损失。
        参数:
        - output: 修复后的图像。
        - target: 目标图像。

        返回:
        - 频域损失值。
        """
        # 快速傅里叶变换 (FFT) 转换到频域
        output_fft = torch.fft.fft2(output, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # 幅值谱
        output_magnitude = torch.abs(output_fft)
        target_magnitude = torch.abs(target_fft)

        # 幅值损失：L1 损失
        magnitude_loss = self.l1_loss(output_magnitude, target_magnitude)

        # (可选) 相位损失：如果需要，可以解开注释
        # output_phase = torch.angle(output_fft)
        # target_phase = torch.angle(target_fft)
        # phase_loss = self.l1_loss(output_phase, target_phase)

        # 仅使用幅值损失（相位损失可根据需求添加）
        return magnitude_loss