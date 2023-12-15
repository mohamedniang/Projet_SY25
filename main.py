import torch
import numpy as np
import os
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
base_path = "images"
processed_path = "images/processed"
img = "images/image2.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

pred = results.xyxy[0].cpu().numpy()

image = cv2.imread(img)

for xyxy in pred:
    label = model.names[int(xyxy[5])]
    if label.lower() == "person":
        cv2.rectangle(
            image,
            (int(xyxy[0]), int(xyxy[1])),
            (int(xyxy[2]), int(xyxy[3])),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            image,
            label,
            (int(xyxy[0]), int(xyxy[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

cv2.imshow("processed image", image)
cv2.waitKey(0)
# filename = os.path.join(processed_path, img)
# print(filename)
# cv2.imwrite(filename, image)