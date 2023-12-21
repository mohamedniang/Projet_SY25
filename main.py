import torch
import numpy as np
import sys
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
def process_image(img_path):
    # Images
    base_path = "images"
    processed_path = "images/processed"
    img = "image1.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    if not img_path : img_path = f"{base_path}/{img}"
    # Inference
    results = model(img_path)

    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    pred = results.xyxy[0].cpu().numpy()

    image = cv2.imread(img_path)

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
    filename = f"{processed_path}/{img}"
    print(filename)
    cv2.imwrite(filename, image)

    return filename


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_string = process_image(input_image_path)
    print(output_string)