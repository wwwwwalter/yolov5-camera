import cv2
import torch
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import ToTensor, Normalize
from torchvision.transforms.v2 import Transform

# 选择设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 加载 yolov5s-cls 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s-cls.pt')

# 将模型移动到合适的设备上
model.to(device)

# 将模型设置为评估模式
model.eval()

# IMAGENET 均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImagePreprocessor(Transform):
    def __init__(self, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        self.transforms = [
            ToTensor(),
            # Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # ToTensor()
        ]

    def forward(self, image):
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # CenterCrop, Resize
        imh, imw = image.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        resized_image = cv2.resize(image[top: top + m, left: left + m], self.target_size, interpolation=cv2.INTER_LINEAR)


        # Apply transforms
        for transform in self.transforms:
            image = transform(resized_image)
            print(type(image))

        # Add batch dimension
        image = image.unsqueeze(0)

        # Move tensor to device
        image = image.to(device)

        return image

def preprocess_image(image, target_size=(224, 224)):
    preprocessor = ImagePreprocessor(target_size=target_size)
    tensor_image = preprocessor(image)
    return tensor_image

def main():
    # 图像路径
    img_path = 'bus.jpg'  # 替换为你的图像路径

    # 使用 OpenCV 读取图像
    img = cv2.imread(img_path)

    # 打印图像的形状
    print("Image shape:", img.shape)

    # 打印图像的信息
    print("Image dtype:", img.dtype)

    # 显示图像
    # cv2.imshow('Image', img)

    # 预处理图片
    tensor_image = preprocess_image(img)

    # 推理
    with torch.no_grad():
        results = model(tensor_image)

    # 解析结果
    probs = torch.softmax(results, dim=1)  # 应用 softmax 函数获取概率分布
    probs_topk, indices_topk = torch.topk(probs, k=5)  # 获取前5个最高概率及其索引
    print(probs_topk)
    print(indices_topk)

    # 获取类别名称
    probs_list = probs_topk[0].tolist()
    indices_list = indices_topk[0].tolist()
    class_names = [model.names[idx] for idx in indices_list]

    # 格式化输出
    category_width = max(len(name) for name in class_names) + 2  # 加2是为了留一些空格
    confidence_width = 10  # 固定宽度，确保置信度对齐
    for class_name, prob in zip(class_names, probs_list):
        print(f"类别: {class_name:<{category_width}} 置信度: {prob:.4f}")

if __name__ == '__main__':
    main()
