import torch
import time

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
# model.cpu()
model.cuda()

model.eval()

# check
device = next(model.parameters()).device
print(f"Model is on device: {device}")

print(type(model))


# Image
im = "https://ultralytics.com/images/zidane.jpg"

with torch.no_grad():
    for i in range(1) :
        # Inference
        t0=time.time()
        results = model(im)
        print((time.time()-t0)*1000)
        print(type(results))

        print("====")
        results.print()
        print("====")
        print(results)
        print("====")
        print(results.pandas().xyxy[0])