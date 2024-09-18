import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load image
img_path = 'small.jpg'  # 替换为你的图像路径
# PIL 打开图像
img = Image.open(img_path)
# 打印img 的形状
print(img.size)
# 打印img的信息
print(img.format, img.mode)
# plt.imshow(img)
# plt.show()

# # 使用OpenCV读取图像
# img = cv2.imread(img_path)
# # 打印图像的形状
# print("Image shape:", img.shape)
# # 打印图像的信息
# print("Image dtype:", img.dtype)
#
# print(img)
# print(type(img))






img = np.array(img)  # 转换为NumPy数组
print(img.dtype)
img = img.astype(np.float32)
print(img.dtype)
img = img / 255.0  # 归一化
print(img)
# img = np.transpose(img, (2, 0, 1))  # HWC to CHW
# print(img)
# img = torch.from_numpy(img).float().unsqueeze(0)  # 转换为Tensor并增加batch维度
# print(img)