import torch

# 检查CUDA是否可用
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"可用GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# 简单的张量计算测试
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x @ y  # 矩阵乘法

print("\nGPU计算测试结果:")
print(z)
print(f"张量是否在GPU上: {z.is_cuda}")