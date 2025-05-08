import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import distance_transform_edt, binary_dilation
from model import InpaintCNN, add_frequency_channels


def inpaint():
    # 定义路径
    image_path = "Azhe.jpg"
    mask_path = "mask.jpg"
    model_path = "image_inpaint_6channel.pth"

    # 检查文件存在性
    if not all(map(lambda x: os.path.exists(x), [image_path, mask_path, model_path])):
        raise FileNotFoundError("请检查 image.jpg、mask.jpg 和 image_inpaint_6channel.pth 是否存在")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 读取图像和掩码
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # 转换为张量
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    mask_np = np.array(mask)
    mask_binary = np.where(mask_np > 128, 1, 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # 添加频域通道
    inputs = add_frequency_channels(image_tensor, mask_tensor).to(device)  # [1, 6, H, W]

    # 生成带掩码的可视化图像
    image_np = np.array(image)
    masked_vis = image_np.copy()
    masked_vis[mask_binary == 0] = 0
    Image.fromarray(masked_vis).save("image_masked.jpg")

    # 推理
    with torch.no_grad():
        outputs = model(inputs)

    # 后处理
    outputs_np = outputs.squeeze().cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    outputs_np = (outputs_np * 0.5 + 0.5) * 255  # 反归一化到 [0, 255]
    outputs_np = np.clip(outputs_np, 0, 255).astype(np.uint8)

    output_image = Image.fromarray(outputs_np)
    output_image.save("output.jpg")

    # 修复 image_masked 和 output 的组合
    combined_image = masked_vis.copy()  # 从 image_masked 开始操作
    black_region = (masked_vis == 0).all(axis=-1)  # 找到 image_masked 的黑色部分
    combined_image[black_region] = outputs_np[black_region]  # 替换黑色部分为 output 的生成内容
    Image.fromarray(combined_image).save("image_combine.jpg")

    # 生成 output_final：外围保留 image_masked，内围逐渐过渡到 output
    def smooth_transition(image_masked, output, mask_binary):
        # 1. 扩大 mask_binary，避免边界采样到黑色区域
        dilated_mask_binary = binary_dilation(mask_binary, structure=np.ones((5, 5)))  # 扩展3-5个像素

        # 2. 计算待修复区域（黑色=0）到白色保留区的距离
        distance = distance_transform_edt(1 - dilated_mask_binary)  # 扩展后的距离计算
        max_distance = np.max(distance)

        # 3. 提取破损边缘的参考颜色（距离=1的像素）
        border_mask = (distance == 1)  # 破损区的边缘像素
        border_colors = image_masked.copy()

        # 进一步过滤边界颜色，确保不采样黑色
        valid_border_mask = border_mask & (np.mean(border_colors, axis=-1) > 0)  # 非黑色像素
        border_colors[~valid_border_mask] = 0  # 只保留有效边缘颜色

        # 4. 计算权重：距离越大，output权重越高
        weights = distance / max_distance  # 线性权重
        weights = np.expand_dims(weights, axis=-1)  # [H, W, 1]

        # 5. 混合逻辑：
        #    - 边缘（distance=1）：完全用 border_colors（image_masked 的边缘颜色）
        #    - 向内过渡：border_colors 和 output 插值
        #    - 中心（distance=max）：完全用 output
        output_final = (1 - weights) * border_colors + weights * output

        # 6. 保留区（mask_binary=1）不受影响
        output_final = np.where(
            np.expand_dims(mask_binary, -1) == 1,
            image_masked,  # 保留区用原图
            output_final  # 待修复区用插值结果
        )

        return output_final.astype(np.uint8)
    # 调用平滑过渡函数
    output_final = smooth_transition(masked_vis, outputs_np, mask_binary)

    # 保存最终结果
    Image.fromarray(output_final).save("output_final.jpg")
    print("修复完成！结果已保存为 output.jpg、image_combine.jpg 和 output_final.jpg")


if __name__ == "__main__":
    inpaint()