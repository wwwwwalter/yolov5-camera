import cv2
import torch
import time


# 加载自定义模型
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# 加载预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 加载 yolov5s 模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # 加载 yolov5m 模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # 加载 yolov5l 模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # 加载 yolov5x 模型

# 设置模型为评估模式
model.eval()

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，也可以指定其他摄像头编号

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    if not ret:
        break

    # 转换为 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.time()

    # 进行推理
    results = model(rgb_frame)

    # 获取预测后的图像数据
    result_image = results.render()[0]

    # rgb2bgr
    bgr_frame = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # results.show()
    # print((time.time()-t0)*1000)

    # # 获取检测结果
    # detections = results.pandas().xyxy[0]
    #
    # # 绘制边界框和标签
    # for _, detection in detections.iterrows():
    #     x1, y1, x2, y2, confidence, class_id, class_name = detection
    #
    #     # 绘制边界框
    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #
    #     # 绘制标签
    #     label = f"{class_name} {confidence:.2f}"
    #     cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # t1 = time.time()-t0
    #
    # print(t1*1000)

    # 显示图像
    cv2.imshow('Object Detection', bgr_frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()



# import cv2
#
# # 1. 打开摄像头
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0 表示默认摄像头，也可以指定其他摄像头编号
#
# # 2. 检查摄像头是否成功打开
# if not cap.isOpened():
#     print("无法打开摄像头，请检查设备连接。")
#     exit()
#
# # 3. 循环读取摄像头帧并显示
# while True:
#     # 读取摄像头帧
#     ret, frame = cap.read()
#
#     if not ret:
#         print("无法获取帧，请检查摄像头。")
#         break
#
#     # 显示图像
#     cv2.imshow('Camera Stream', frame)
#
#     # 按 'q' 键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 4. 释放摄像头资源并关闭窗口
# cap.release()
# cv2.destroyAllWindows()



