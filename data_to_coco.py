# This script coverts the https://www.kaggle.com/datasets/pkdarabi/cardetection/data dataset to the COCO format.

import kagglehub
import os
import json
from PIL import Image

data_path = kagglehub.dataset_download("pkdarabi/cardetection")
data_path = os.path.join(data_path, "car")
class_names = [
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
]
class_map = {i: name for i, name in enumerate(class_names)}

for set in ["train", "test", "valid"]:
    image_dir = os.path.join(data_path, f"{set}/images")
    label_dir = os.path.join(data_path, f"{set}/labels")
    output_json = os.path.join(data_path, f"{set}/cocoann.json")

    # Prepare base structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in class_map.items()],
    }

    annotation_id = 1
    image_id = 1

    for filename in os.listdir(image_dir):
        if not filename.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        # Load image size
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append(
            {"id": image_id, "file_name": image_path, "width": width, "height": height}
        )

        # Load label
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    x = (x_center - w / 2) * width
                    y = (y_center - h / 2) * height
                    w = w * width
                    h = h * height

                    coco["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

        image_id += 1

    # Write COCO JSON
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO annotation saved to {output_json}")
