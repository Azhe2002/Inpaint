import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import InpaintCNN, add_frequency_channel
from loss import InpaintLoss
import os
import numpy as np


class InpaintDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                          if f.lower().endswith(('.jpg', '.jpeg'))]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                           if f.lower().endswith(('.jpg', '.jpeg'))]

        # 验证数据匹配
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError("图像与掩码数量不匹配")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 加载图像
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)

        # 加载并处理Mask（单通道二值化）
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask_np = np.array(mask)
        mask_binary = np.where(mask_np > 128, 1, 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        # 返回图像和掩码
        return img, mask_tensor, img  # 目标为完整图像


# 训练参数
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001


def train():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = InpaintLoss(use_frequency_loss=True, lambda_freq=0.1)  # 启用频域损失

    try:
        # 加载数据
        dataset = InpaintDataset("data/images", "data/masks")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        print(f"训练开始，共 {len(dataset)} 对图像数据")
        print(f"批次大小: {BATCH_SIZE}, 总迭代次数: {EPOCHS}")

        # 训练循环
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for batch_idx, (images, masks, targets) in enumerate(dataloader):
                images, masks, targets = images.to(device), masks.to(device), targets.to(device)

                # 合并输入（添加频域通道）
                inputs = add_frequency_channel(images, masks)

                optimizer.zero_grad()
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, targets, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 每10个batch打印一次进度
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{EPOCHS}], 平均Loss: {avg_loss:.4f}")

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        return

    # 保存模型
    torch.save(model.state_dict(), "image_inpaint.pth")
    print("训练完成，模型已保存为 image_inpaint.pth")


if __name__ == "__main__":
    train()