import cv2
import torch
import pandas
from torchvision.transforms.v2.functional import resize

# IMAGENET 均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# choose cuda or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 加载 yolov5s-cls 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s-cls.pt')

# 将模型移动到合适的设备上
model.to(device)

# 将模型设置为评估模式
model.eval()


def preprocess_image_0(image, target_size=(224, 224)):
    """
    预处理图像，使其符合模型输入要求。
    """
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    imh, imw = image.shape[:2]
    m = min(imh, imw)  # min dimension
    top, left = (imh - m) // 2, (imw - m) // 2
    resized_image = cv2.resize(image[top: top + m, left: left + m], target_size, interpolation=cv2.INTER_LINEAR)

    # 将图像从 [0, 255] 转换到 [0, 1]
    image_normalized = resized_image.astype('float32') / 255.0

    # 减去均值
    image_centered = image_normalized - IMAGENET_MEAN

    # 除以标准差
    normalized_image = image_centered / IMAGENET_STD

    # 确保结果为 float32 类型
    normalized_image = normalized_image.astype('float32')

    # numpy to tensor, HWC to CHW, add batch dimension
    tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0)

    # tensor to device
    tensor_image = tensor_image.to(device)
    return tensor_image


def preprocess_image_1(image, target_size=(224, 224)):
    """
    预处理图像，使其符合模型输入要求。
    """
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    imh, imw = image.shape[:2]
    m = min(imh, imw)  # min dimension
    top, left = (imh - m) // 2, (imw - m) // 2
    resized_image = cv2.resize(image[top: top + m, left: left + m], target_size, interpolation=cv2.INTER_LINEAR)

    # HWC to CHW
    resized_image = resized_image.transpose((2, 0, 1))
    # 归一化
    normalized_image = resized_image.astype('float32') / 255.0
    # numpy to tensor, add batch dimension
    tensor_image = torch.from_numpy(normalized_image.astype('float32')).unsqueeze(0)
    # tensor to device
    tensor_image = tensor_image.to(device)
    return tensor_image


def preprocess_image_2(image, target_size=(224, 224)):
    """
    预处理图像，使其符合模型输入要求。
    """
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    resized_image = cv2.resize(image, target_size)
    # # uint8 to float32, 0-255 -> 0-1
    # normalized_image = resized_image.astype('float32') / 255.0
    # uint8 to float32, 0-255 -> 0-1
    normalized_image = (resized_image.astype('float32') / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    print(normalized_image.dtype)
    # numpy to tensor, HWC to CHW, add batch dimension
    tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0)
    # tensor to device
    tensor_image = tensor_image.to(device)
    return tensor_image


def main():
    # 图像路径
    img_path = '../data/images/girl.png'  # 替换为你的图像路径

    # 使用OpenCV读取图像
    img = cv2.imread(img_path)

    # 打印图像的形状
    print("Image shape:", img.shape)

    # 打印图像的信息
    print("Image dtype:", img.dtype)

    # 显示图像
    # cv2.imshow('Image', img)

    # 预处理图片
    tensor_image = preprocess_image_0(img)
    #
    # 推理
    with torch.no_grad():
        results = model(tensor_image)

    # print(results)
    print(results.shape)
    # 解析结果
    probs = torch.softmax(results, dim=1)  # 应用 softmax 函数获取概率分布

    # print(probs)
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
        print(f"类别: {class_name:<{category_width}} 置信度: {prob:.2f}")




if __name__ == '__main__':
    main()
