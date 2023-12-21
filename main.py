import torch
import numpy as np
import os
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
base_path = "images"
processed_path = "images/processed"
img = "image1.jpg"  # or file, Path, PIL, OpenCV, numpy, list
img_path = f"{base_path}/{img}"
# Inference
results = model(img_path)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

pred = results.xyxy[0].cpu().numpy()

<<<<<<< HEAD
image = cv2.imread(img_path)
=======
def count_person(results):
    pred = results.xyxy[0].cpu().numpy()

    # count the number of person
    person_count = sum(1 for xyxy in pred if model.names[int(xyxy[5])].lower() == 'person')

    return 'this image has %s person.' %person_count

image = cv2.imread(img)
>>>>>>> 21012ce (create a new function of counting the person)

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
<<<<<<< HEAD
filename = f"{processed_path}/{img}"
print(filename)
cv2.imwrite(filename, image)
=======
filename = os.path.join(processed_path, img)
print(filename)
cv2.imwrite(filename, image)
>>>>>>> 21012ce (create a new function of counting the person)
