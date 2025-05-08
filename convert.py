import os
from PIL import Image


def convert_images_to_grayscale(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 检查文件是否为图片
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 打开图像
                img = Image.open(file_path)

                # 确认图像尺寸为 128x128x3
                if img.size == (128, 128) and img.mode == 'RGB':
                    # 转换为灰度图像
                    gray_img = img.convert('L')

                    # 保存覆盖原文件
                    gray_img.save(file_path)
                    print(f"已转换并覆盖: {filename}")
                else:
                    print(f"跳过文件（不符合尺寸或模式）: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 指定目录路径
directory_path = './masks'

# 执行转换
convert_images_to_grayscale(directory_path)