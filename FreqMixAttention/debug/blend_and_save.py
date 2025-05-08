import cv2
import numpy as np

def smooth_image(combine_path, mask_path, output_path):
    # 读取 combine 和 mask
    combine = cv2.imread(combine_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if combine is None or mask is None:
        raise FileNotFoundError("Combine or Mask image not found.")

    # 确保图像尺寸一致
    if combine.shape[:2] != mask.shape[:2]:
        raise ValueError("Combine and Mask must have the same dimensions.")

    # 将 mask 归一化到 [0, 1] 范围
    mask_normalized = mask / 255.0

    # 计算平滑区域的权重
    weights = cv2.distanceTransform((mask_normalized < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    weights = cv2.normalize(weights, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 扩展 mask 的维度以匹配 combine 的通道数 (3通道)
    mask_3d = np.repeat(mask_normalized[:, :, np.newaxis], 3, axis=2)
    weights_3d = np.repeat(weights[:, :, np.newaxis], 3, axis=2)

    # 平滑处理: 根据 mask 和 weights 渐变过渡
    final_image = combine * mask_3d + (1 - mask_3d) * combine * weights_3d

    # 将结果保存为 final.jpg
    cv2.imwrite(output_path, final_image.astype(np.uint8))
    print(f"Final image saved to {output_path}")

if __name__ == "__main__":
    # 使用示例
    smooth_image("image_combine.jpg", "mask.jpg", "final.jpg")