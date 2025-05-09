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
from functools import partial
from dataclasses import dataclass
import argparse


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
        dict: A dictionary mapping image IDs to their sizes (height, width).
    """
    coco = COCO(cocoann_file)
    return {img["id"]: (img["height"], img["width"]) for img in coco.dataset["images"]}


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def compute_metrics(eval_pred, image_processor, cocoann_file, size_map):
    """
    Compute metrics for evaluation using COCOeval.
    Args:
        eval_pred: Tuple of predictions and labels.
        image_processor: Image processor for post-processing.
        cocoann_file: Path to the COCO annotations file.
        size_map: Dictionary mapping image IDs to their sizes.
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    image_sizes = []
    for batch in labels:
        batch_image_sizes = [size_map[int(lbl["image_id"])] for lbl in batch]
        image_sizes.append(batch_image_sizes)

    processed_predictions = []
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(
            logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
        )
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=0.01, target_sizes=target_sizes
        )
        processed_predictions.extend(post_processed_output)

    processed_labels = []
    for batch in labels:
        for label in batch:
            processed_labels.append({"image_id": label["image_id"]})

    # Format the predictions to COCO format
    coco_predictions = []
    for i, output in enumerate(processed_predictions):
        for box, score, label in zip(
            output["boxes"], output["scores"], output["labels"]
        ):
            x1, y1, x2, y2 = box.tolist()
            coco_predictions.append(
                {
                    "image_id": int(processed_labels[i]["image_id"]),
                    "category_id": int(label),
                    "bbox": [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                    ],  # COCO bbox format: [x, y, w, h]
                    "score": float(score),
                }
            )

    # Use COCOeval for evaluation
    coco_gt = COCO(annotation_file=cocoann_file)  # Ground truth
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


def main(args, config):
    if args.clear:
        # Clear previous outputs
        if os.path.exists(config["output_dir"]):
            os.system(f"rm -rf {config['output_dir']}")
        if os.path.exists(config["logging_dir"]):
            os.system(f"rm -rf {config['logging_dir']}")
        print("Cleared logs and checkpoints")

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

    size_map = create_size_map(config["valid_ann"])

    # Set up training arguments
    output_dir = os.path.join(config["output_dir"], os.path.basename(checkpoint))
    training_args = TrainingArguments(
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        output_dir=output_dir,
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        logging_dir=os.path.join(config["logging_dir"], os.path.basename(checkpoint)),
        report_to=["tensorboard"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mAP",
        greater_is_better=True,
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        fp16_full_eval=config["fp16_full_eval"],
        remove_unused_columns=False,
        eval_do_concat_batches=False,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=partial(
            compute_metrics,
            image_processor=image_processor,
            cocoann_file=config["valid_ann"],
            size_map=size_map,
        ),
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

    # Save the model
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    print(f"Model saved to {best_model_dir}")

    # Evaluate on test set
    if "test_ann" in config:
        test_dataset = COCODataset(
            coco_json_path=config["test_ann"],
            image_root=config["test_img"],
            image_processor=image_processor,
        )

        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print("Final evaluation on test set:\n", results)
    else:
        results = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="test")
        print("Final evaluation on validation set:\n", results)

    print("Training and evaluation completed.")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear", action="store_true", help="Clear logs and checkpoints"
    )
    return parser.parse_args()


if __name__ == "__main__":
    config = load_config(config_path="config.yaml")
    args = parse_args()
    main(args, config)
