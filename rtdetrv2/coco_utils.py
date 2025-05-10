import os
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class COCODataset(Dataset):
    """
    Custom dataset class for loading COCO format datasets.
    Args:
        cocoann_file: Path to the COCO annotations file.
        image_processor: Image processor for preprocessing images.
        image_root: Directory containing the images. Can be omitted if the annotation file contains full paths to the images.
        transforms: Albumentations transforms for data augmentation.
    """

    def __init__(self, cocoann_file, image_processor, image_root="", transforms=None):
        with open(cocoann_file, "r") as f:
            coco = json.load(f)

        self.image_root = image_root
        self.image_processor = image_processor
        self.transforms = transforms

        # Map image_id to image info
        self.images = {img["id"]: img for img in coco["images"]}

        # Group annotations by image_id
        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        anns = self.annotations.get(image_id, [])

        img_path = os.path.join(self.image_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Apply augmentations
        if self.transforms:
            image = np.array(image)
            bboxes = [ann["bbox"] for ann in anns]
            category_ids = [ann["category_id"] for ann in anns]

            augmented = self.transforms(
                image=image, bboxes=bboxes, category_ids=category_ids
            )

            image = Image.fromarray(augmented["image"])
            # Reconstruct anns from transformed bboxes
            anns = [
                {
                    "bbox": bbox,
                    "category_id": cids,
                    "image_id": image_id,
                    "area": float(bbox[2] * bbox[3]),
                }
                for bbox, cids in zip(augmented["bboxes"], augmented["category_ids"])
            ]

        # Apply image_processor
        encoding = self.image_processor(
            images=image,
            annotations={"image_id": image_id, "annotations": anns},
            return_tensors="pt",
        )

        encoding["pixel_values"] = encoding["pixel_values"].squeeze(0)
        encoding["labels"] = encoding["labels"][0]
        encoding["labels"]["image_id"] = encoding["labels"]["image_id"].squeeze(0)

        return encoding


def create_size_map(cocoann_file):
    """
    Creates a size map from the COCO annotations file.
    Args:
        cocoann_file: Path to the COCO annotations file.
    Returns:
        dict: A dictionary mapping image IDs to the image sizes (height, width).
    """
    coco = COCO(cocoann_file)
    return {img["id"]: (img["height"], img["width"]) for img in coco.dataset["images"]}


def get_classes_from_coco(cocoann_file):
    """
    Get class names from the COCO annotations file.
    Args:
        cocoann_file: Path to the COCO annotations file.
    Returns:
        list: A list of class names.
    """
    coco = COCO(cocoann_file)
    cats = sorted(coco.dataset["categories"], key=lambda c: c["id"])
    classes = [c["name"] for c in cats]
    return classes
