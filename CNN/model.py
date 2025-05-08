import torch.nn as nn


class InpaintCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入通道改为4 (RGB + Mask)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, padding=2),  # 修改输入通道为4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # 解码器保持不变
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
