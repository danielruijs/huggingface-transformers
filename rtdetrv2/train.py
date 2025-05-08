from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import Dataset
import yaml
import os
import json
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODataset(Dataset):
    """
    Custom dataset class for loading COCO format datasets.
    Args:
        coco_json_path (str): Path to the COCO JSON file.
        image_root (str): Directory containing the images.
        image_processor: Image processor for preprocessing images.
    """

    def __init__(self, coco_json_path, image_root, image_processor):
        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        self.image_root = image_root
        self.image_processor = image_processor

        # Map image_id to file_name
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

        # Get class labels and boxes
        class_labels = [ann["category_id"] for ann in anns]
        boxes = [ann["bbox"] for ann in anns]  # COCO bbox format: [x, y, w, h]

        # Apply image_processor (e.g., DetrImageProcessor)
        encoding = self.image_processor(
            images=image,
            annotations={"image_id": image_id, "annotations": anns},
            return_tensors="pt",
        )

        for k, v in encoding.items():
            if hasattr(v, "squeeze"):
                encoding[k] = v.squeeze()

        encoding["labels"] = {
            "class_labels": torch.tensor(class_labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }

        return encoding


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation using COCOeval.
    Args:
        eval_pred: Tuple of predictions and labels.
    """
    predictions, labels = eval_pred

    # Format the predictions to COCO format
    coco_predictions = []
    for i, (boxes, scores, labels_pred) in enumerate(
        zip(predictions["boxes"], predictions["scores"], predictions["labels"])
    ):
        for box, score, label in zip(boxes, scores, labels_pred):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            coco_predictions.append(
                {
                    "image_id": predictions["image_ids"][i],
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )

    # Use COCOeval for evaluation
    coco_gt = COCO(annotations=labels)  # Ground truth
    coco_dt = coco_gt.loadRes(coco_predictions)  # Predictions
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return metrics
    return {
        "mAP": float(coco_eval.stats[0]),  # mAP@[.5:.95]
        "mAP_50": float(coco_eval.stats[1]),  # mAP@.50
        "mAP_75": float(coco_eval.stats[2]),  # mAP@.75
        "mAP_small": float(coco_eval.stats[3]),  # mAP@[.5:.95] small objects
        "mAP_medium": float(coco_eval.stats[4]),  # mAP@[.5:.95] medium objects
        "mAP_large": float(coco_eval.stats[5]),  # mAP@[.5:.95] large objects
    }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in the batch.
    Args:
        batch: List of dictionaries containing pixel values and labels.
    """
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }


def main(config):
    # Load image processor and pre-trained model
    checkpoint = config["checkpoint"]
    classes = config["classes"]
    image_processor = RTDetrImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": config["image_width"], "height": config["image_height"]},
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(
        checkpoint, num_labels=len(classes), ignore_mismatched_sizes=True
    )

    # Load the datasets
    train_dataset = COCODataset(
        coco_json_path=config["train_ann"],
        image_root=config["train_img"],
        image_processor=image_processor,
    )

    val_dataset = COCODataset(
        coco_json_path=config["valid_ann"],
        image_root=config["valid_img"],
        image_processor=image_processor,
    )

    test_dataset = COCODataset(
        coco_json_path=config["test_ann"],
        image_root=config["test_img"],
        image_processor=image_processor,
    )

    # Set up training arguments
    output_dir = config["output_dir"]
    training_args = TrainingArguments(
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        output_dir=output_dir,
        logging_dir=config["logging_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mAP",
        greater_is_better=True,
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        fp16_full_eval=config["fp16_full_eval"],
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(os.path.join(output_dir, "final_model"))
    print(f"Model saved to {os.path.join(output_dir, 'final_model')}")

    # Evaluate on test set
    results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(results)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config(config_path="config.yaml")
    main(config)
