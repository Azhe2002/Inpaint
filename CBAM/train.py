import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from model import InpaintCNN  # 从model.py导入
from loss import InpaintLoss   # 从loss.py导入


# 数据集类定义（直接内嵌在train函数上方）
class InpaintDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"图像数量({len(self.img_paths)})与掩码数量({len(self.mask_paths)})不匹配")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 加载并转换图像
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)

        # 加载并二值化掩码
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(
            np.where(mask_np > 128, 1.0, 0.0)
        ).float().unsqueeze(0)

        # 合成4通道输入 [RGB + Mask]
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
        return input_tensor, img_tensor  # (input, target)

def train():
    # 训练配置
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 0.001
    DATA_DIR = "data"
    MODEL_SAVE_PATH = "image_inpaint.pth"

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")


    # 初始化模型、损失函数和优化器
    model = InpaintCNN().to(device)
    criterion = InpaintLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 数据加载
    try:
        dataset = InpaintDataset(
            img_dir=os.path.join(DATA_DIR, "images"),
            mask_dir=os.path.join(DATA_DIR, "masks")
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        print(f"\n开始训练，数据量: {len(dataset)}")
        print(f"批次大小: {BATCH_SIZE}, 总epoch数: {EPOCHS}\n")

        # 训练循环
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(dataloader, 1):
                inputs, targets = inputs.to(device), targets.to(device)
                masks = inputs[:, 3:, :, :]  # 提取mask通道

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 打印批次进度
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        f"Epoch: {epoch:03d}/{EPOCHS} | "
                        f"Batch: {batch_idx:03d}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.2e}"
                    )

            # 打印epoch统计
            avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch: {epoch:03d} 平均Loss: {avg_loss:.4f}\n{'-'*50}")

    except Exception as e:
        print(f"\n训练出错: {str(e)}\n")
        return

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n训练完成，模型已保存至: {MODEL_SAVE_PATH}\n")

if __name__ == "__main__":
    train()