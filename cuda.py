import torch

# 创建张量 A
A = torch.randn(3, 4)

# 创建张量 B
B = torch.randn(4)

# 执行除法
result = B / A

# 输出结果的形状
print(result.shape)  # 应该输出 (3, 4)