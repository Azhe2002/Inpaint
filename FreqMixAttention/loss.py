import torch
import torch.nn as nn
import torch.fft


class InpaintLoss(nn.Module):
    def __init__(self, use_frequency_loss=True, use_color_consistency_loss=True, lambda_freq=0.1, lambda_color=0.1):
        """
        初始化损失函数。
        参数:
        - use_frequency_loss: 是否启用频域损失。
        - use_color_consistency_loss: 是否启用颜色一致性损失。
        - lambda_freq: 频域损失的权重系数。
        - lambda_color: 颜色一致性损失的权重系数。
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.use_frequency_loss = use_frequency_loss
        self.use_color_consistency_loss = use_color_consistency_loss
        self.lambda_freq = lambda_freq
        self.lambda_color = lambda_color

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

        # 频域损失
        freq_loss = 0
        if self.use_frequency_loss:
            freq_loss = self.compute_frequency_loss(output, target)

        # 颜色一致性损失
        color_loss = 0
        if self.use_color_consistency_loss:
            color_loss = self.compute_color_consistency_loss(output, target, mask)

        # 总损失
        total_loss = spatial_loss + self.lambda_freq * freq_loss + self.lambda_color * color_loss
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

        # 相位谱
        output_phase = torch.angle(output_fft)
        target_phase = torch.angle(target_fft)

        # 相位损失：L1 损失
        phase_loss = self.l1_loss(output_phase, target_phase)

        # 频域总损失 = 幅值损失 + 相位损失
        return magnitude_loss + phase_loss

    def compute_color_consistency_loss(self, output, target, mask):
        """
        计算颜色一致性损失。
        参数:
        - output: 修复后的图像。
        - target: 目标图像。
        - mask: 掩码。

        返回:
        - 颜色一致性损失值。
        """
        # 提取修复区域
        repaired_region = output * (1 - mask)
        target_region = target * (1 - mask)

        # 统计均值和标准差
        output_mean, output_std = repaired_region.mean(dim=(2, 3)), repaired_region.std(dim=(2, 3))
        target_mean, target_std = target_region.mean(dim=(2, 3)), target_region.std(dim=(2, 3))

        # 颜色一致性损失：均值和标准差的差异
        mean_loss = torch.abs(output_mean - target_mean).mean()
        std_loss = torch.abs(output_std - target_std).mean()

        return mean_loss + std_loss