import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image


def compare_images(output1_path, output2_path, reference_path):
    """
    比较两个修复结果与参考图像的效果。

    参数:
    - output1_path: 修复结果 1 的路径。
    - output2_path: 修复结果 2 的路径。
    - reference_path: 原始未破损图像的路径。

    返回:
    - 一个包含 SSIM 和 PSNR 的对比结果，并可视化得分。
    """
    # 读取图像
    output1 = np.array(Image.open(output1_path).convert("RGB"))
    output2 = np.array(Image.open(output2_path).convert("RGB"))
    reference = np.array(Image.open(reference_path).convert("RGB"))

    # 检查图像尺寸是否一致
    if output1.shape != reference.shape or output2.shape != reference.shape:
        raise ValueError("输出图像和参考图像的尺寸必须一致")

    # 确保 win_size 不超过图像最小边
    min_dim = min(reference.shape[0], reference.shape[1])
    win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)

    # 计算 SSIM 和 PSNR
    ssim1 = ssim(output1, reference, channel_axis=-1, win_size=win_size)
    ssim2 = ssim(output2, reference, channel_axis=-1, win_size=win_size)
    psnr1 = psnr(output1, reference, data_range=255)
    psnr2 = psnr(output2, reference, data_range=255)

    # 打印比较结果
    print(f"Output1 - SSIM: {ssim1:.4f}, PSNR: {psnr1:.2f} dB")
    print(f"Output2 - SSIM: {ssim2:.4f}, PSNR: {psnr2:.2f} dB")

    # 可视化比较
    labels = ["SSIM", "PSNR"]
    values1 = [ssim1, psnr1]
    values2 = [ssim2, psnr2]

    x = np.arange(len(labels))  # 横轴位置
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width / 2, values1, width, label="Output_CNN")
    bar2 = ax.bar(x + width / 2, values2, width, label="Output_FMA")

    # 添加标签
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")
    ax.set_title("Comparison-Azhe of Output_CNN and Output_FMA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(bar1, fmt="%.2f", padding=3)
    ax.bar_label(bar2, fmt="%.2f", padding=3)

    # 显示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 替换为实际图像路径
    output1_path = "output_FMA_Azhe_F.jpg"
    output2_path = "output_FMA_Azhe_F2.jpg"
    reference_path = "Azhe.jpg"

    compare_images(output1_path, output2_path, reference_path)