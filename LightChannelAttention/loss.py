import torch.nn as nn

class InpaintLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, output, target, mask):
        # Mask为0表示破损区域（需要修复）
        # 计算破损区域的L1损失
        loss = self.l1_loss(output * (1 - mask), target * (1 - mask))
        return loss