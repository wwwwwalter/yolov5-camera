import cv2

# 图像路径
img_path = 'zidane.jpg'  # 替换为你的图像路径

# 使用OpenCV读取图像
img = cv2.imread(img_path)

# 打印图像的形状
print("Image shape:", img.shape)

# 打印图像的信息
print("Image dtype:", img.dtype)

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)  # 等待按键，0表示无限等待
cv2.destroyAllWindows()  # 销毁所有窗口
