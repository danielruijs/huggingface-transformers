from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import torch
import yaml
import os
import argparse
import albumentations as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from functools import partial
from dataclasses import dataclass
from coco_utils import COCODataset, create_size_map, get_classes_from_coco


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def compute_metrics(eval_pred, image_processor, cocoann_file):
    """
    Compute metrics for evaluation using COCOeval.
    Args:
        eval_pred: Tuple of predictions and labels.
        image_processor: Image processor for post-processing.
        cocoann_file: Path to the COCO annotations file.
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    size_map = create_size_map(cocoann_file=cocoann_file)
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


class AugmentationSwitcher(TrainerCallback):
    def __init__(self, dataset, num_train_epochs, no_aug_epochs):
        self.dataset = dataset
        self.num_train_epochs = num_train_epochs
        self.no_aug_epochs = no_aug_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        if (
            state.epoch >= (self.num_train_epochs - self.no_aug_epochs)
            and self.dataset.transforms is not None
        ):
            print("Disabling augmentations")
            self.dataset.transforms = None
        print("Transforms:", self.dataset.transforms)


def main(args, config):
    output_dir = config["output_dir"]
    logging_dir = config["logging_dir"]
    if args.clear:
        # Clear previous outputs
        if os.path.exists(output_dir):
            os.system(f"rm -rf {output_dir}")
        if os.path.exists(logging_dir):
            os.system(f"rm -rf {logging_dir}")
        print("Cleared logs and checkpoints")

    # Get classes from COCO annotations
    classes = get_classes_from_coco(config["train_ann"])

    # Load image processor and pre-trained model
    checkpoint = config["checkpoint"]
    image_processor = RTDetrImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": config["image_width"], "height": config["image_height"]},
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(
        checkpoint, num_labels=len(classes), ignore_mismatched_sizes=True
    )

    # Load the training dataset
    train_dataset = COCODataset(
        cocoann_file=config["train_ann"],
        image_processor=image_processor,
        image_root=config.get("train_img", ""),
        transforms=build_transforms(config),
    )

    # Load the validation dataset
    val_dataset = COCODataset(
        cocoann_file=config["valid_ann"],
        image_processor=image_processor,
        image_root=config.get("valid_img", ""),
    )

    # Set up training arguments
    output_dir_run = os.path.join(output_dir, os.path.basename(checkpoint), args.name)
    logging_dir_run = os.path.join(logging_dir, os.path.basename(checkpoint), args.name)
    training_args = TrainingArguments(
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        output_dir=output_dir_run,
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        logging_dir=logging_dir_run,
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
        ),
        data_collator=collate_fn,
        callbacks=[
            AugmentationSwitcher(
                train_dataset, config["num_train_epochs"], config["no_aug_epochs"]
            )
        ],
    )

    # Start training
    trainer.train()

    # Save the model and image processor
    best_model_dir = os.path.join(output_dir_run, "best_model")
    trainer.save_model(best_model_dir)
    image_processor.save_pretrained(best_model_dir)
    print(f"Model and image processor saved to {best_model_dir}")

    # Evaluate on test set
    if "test_ann" in config:
        test_dataset = COCODataset(
            cocoann_file=config["test_ann"],
            image_processor=image_processor,
            image_root=config.get("test_img", ""),
        )
        trainer.compute_metrics = partial(
            compute_metrics,
            image_processor=image_processor,
            cocoann_file=config["test_ann"],
        )

        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print("Final evaluation on test set:\n", results)
    else:
        results = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="test")
        print("Final evaluation on validation set:\n", results)

    print("Training and evaluation completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--clear", action="store_true", help="Clear logs and checkpoints"
    )
    parser.add_argument("--name", type=str, default="", help="Name of the experiment")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(config):
    transforms_list = []
    for aug in config.get("augmentations", []):
        aug_class = getattr(A, aug["name"])
        transforms_list.append(aug_class(**aug["params"]))
    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )


if __name__ == "__main__":
    args = parse_args()
    config = load_config(config_path=args.config)
    main(args, config)
