import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import InpaintCNN


def inpaint():
    # 定义路径
    image_path = "image.jpg"
    mask_path = "mask.jpg"
    model_path = "image_inpaint.pth"

    # 检查文件存在性
    if not all(os.path.exists(p) for p in [image_path, mask_path, model_path]):
        missing = [p for p in [image_path, mask_path, model_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"缺失文件: {', '.join(missing)}")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = InpaintCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 读取图像和Mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # 将Mask二值化并调整维度
    mask_np = np.array(mask)
    mask_binary = np.where(mask_np > 128, 1, 0).astype(np.float32)

    # 转换为张量并调整维度（重要修改）
    image_tensor = transform(image).unsqueeze(0)  # 维度变为 [1, 3, H, W]
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).float()  # 维度变为 [1, 1, H, W]

    # 合并输入（维度对齐）
    model_input = torch.cat([image_tensor, mask_tensor], dim=1).to(device)  # 最终维度 [1, 4, H, W]

    # 生成带掩码的可视化图像
    image_np = np.array(image)
    masked_vis = image_np.copy()
    masked_vis[mask_binary == 0] = 0
    Image.fromarray(masked_vis).save("image_masked.jpg")

    # 推理
    with torch.no_grad():
        output = model(model_input)

    # 后处理
    output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_np = (output_np * 0.5 + 0.5) * 255
    final_output = image_np.copy()
    final_output[mask_binary == 0] = output_np[mask_binary == 0]

    Image.fromarray(final_output.astype(np.uint8)).save("output.jpg")
    print("修复完成！生成文件：image_masked.jpg 和 output.jpg")


if __name__ == "__main__":
    inpaint()