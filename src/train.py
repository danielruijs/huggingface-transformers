from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import torch
import yaml
import json
import os
import argparse
import datetime
import albumentations as A
from functools import partial
from dataclasses import dataclass
from coco_utils import (
    COCODataset,
    create_size_map,
    get_classes_from_coco,
    compute_COCO_metrics,
)


EVAL_BATCHES_JSONL = "eval_batches.jsonl"


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
    processed_labels = []
    for batch in labels:
        batch_image_sizes = [size_map[int(lbl["image_id"])] for lbl in batch]
        image_sizes.append(batch_image_sizes)
        for label in batch:
            processed_labels.append({"image_id": label["image_id"]})

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

    return compute_COCO_metrics(
        predictions=processed_predictions,
        labels=processed_labels,
        cocoann_file=cocoann_file,
    )


def serialize_tensor_dict(tensor_dict):
    """
    Convert a dictionary of tensors to a serializable format.
    Args:
        tensor_dict: Dictionary containing tensors.
    Returns:
        dict: A dictionary with tensors converted to lists.
    """
    serializable = {}
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.cpu().tolist()
        else:
            serializable[k] = v
    return serializable


def compute_metrics_batch(
    eval_pred, image_processor, cocoann_file, size_map, compute_result
):
    """
    Batch-wise compute_metrics that writes each batch to disk, then on compute_result=True does the full COCO evaluation.
    Args:
        eval_pred: Tuple of predictions and labels.
        image_processor: Image processor for post-processing.
        cocoann_file: Path to the COCO annotations file.
        compute_result: Boolean flag to compute final results.
    Returns:
        dict: A dictionary containing evaluation metrics if compute_result is True.
    """
    # During batch calls: process & append to disk, return empty
    if not compute_result:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        batch_image_sizes = [size_map[int(lbl["image_id"])] for lbl in labels]

        output = ModelOutput(
            logits=predictions[1].detach().clone(),
            pred_boxes=predictions[2].detach().clone(),
        )
        post = image_processor.post_process_object_detection(
            output, threshold=0.01, target_sizes=batch_image_sizes
        )

        # Save predictions and labels to JSONL file
        with open(EVAL_BATCHES_JSONL, "a") as f:
            for det, lbl in zip(post, labels):
                record = {
                    "predictions": serialize_tensor_dict(det),
                    "labels": serialize_tensor_dict(lbl),
                }
                f.write(json.dumps(record) + "\n")

        return {}

    # Final call: load predictions and labels and compute COCO metrics
    else:
        all_preds = []
        all_labels = []
        with open(EVAL_BATCHES_JSONL, "r") as f:
            for line in f:
                rec = json.loads(line)
                all_preds.append(rec["predictions"])
                all_labels.append(rec["labels"])

        metrics = compute_COCO_metrics(
            predictions=all_preds,
            labels=all_labels,
            cocoann_file=cocoann_file,
        )

        os.remove(EVAL_BATCHES_JSONL)
        return metrics


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
        print("\nTransforms:", self.dataset.transforms)


def main(args, config):
    checkpoint = config["checkpoint"]

    if os.path.exists(EVAL_BATCHES_JSONL):
        os.remove(EVAL_BATCHES_JSONL)

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, config["output_dir"])
    logging_dir = os.path.join(script_dir, config["logging_dir"])
    train_ann = os.path.join(script_dir, config["train_ann"])
    valid_ann = os.path.join(script_dir, config["valid_ann"])
    test_ann = os.path.join(script_dir, config.get("test_ann", ""))
    train_img = (
        os.path.join(script_dir, config["train_img"]) if "train_img" in config else ""
    )
    valid_img = (
        os.path.join(script_dir, config["valid_img"]) if "valid_img" in config else ""
    )
    test_img = (
        os.path.join(script_dir, config["test_img"]) if "test_img" in config else ""
    )

    output_dir_run = os.path.join(output_dir, os.path.basename(checkpoint), args.name)
    logging_dir_run = os.path.join(logging_dir, os.path.basename(checkpoint), args.name)
    best_model_dir = os.path.join(output_dir_run, "best_model")

    if args.clear:
        # Clear previous outputs
        if os.path.exists(output_dir):
            os.system(f"rm -rf {output_dir}")
        if os.path.exists(logging_dir):
            os.system(f"rm -rf {logging_dir}")
        print("Cleared logs and checkpoints")

    # Get classes from COCO annotations
    classes = get_classes_from_coco(train_ann)

    # Load image processor and pre-trained model
    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": config["image_width"], "height": config["image_height"]},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint, num_labels=len(classes), ignore_mismatched_sizes=True
    )
    model.config.id2label = {i: cls for i, cls in enumerate(classes)}

    # Save the image processor
    image_processor.save_pretrained(best_model_dir)

    # Load the training dataset
    train_dataset = COCODataset(
        cocoann_file=train_ann,
        image_processor=image_processor,
        image_root=train_img,
        transforms=build_transforms(config),
    )

    # Load the validation dataset
    val_dataset = COCODataset(
        cocoann_file=valid_ann,
        image_processor=image_processor,
        image_root=valid_img,
    )

    batch_eval_metrics = config.get("batch_eval_metrics", False)
    if batch_eval_metrics:
        compute_metrics_func = partial(
            compute_metrics_batch,
            image_processor=image_processor,
            cocoann_file=valid_ann,
            size_map=create_size_map(cocoann_file=valid_ann),
        )
    else:
        compute_metrics_func = partial(
            compute_metrics,
            image_processor=image_processor,
            cocoann_file=valid_ann,
        )

    # Set up training arguments
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
        save_safetensors=config.get("save_safetensors", True),
        batch_eval_metrics=batch_eval_metrics,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_func,
        data_collator=collate_fn,
        callbacks=[
            AugmentationSwitcher(
                train_dataset, config["num_train_epochs"], config["no_aug_epochs"]
            )
        ],
    )

    # Start training
    trainer.train()

    # Save best model
    trainer.save_model(best_model_dir)
    print(f"Best model saved to {best_model_dir}")

    # Evaluate on test set
    if "test_ann" in config:
        test_dataset = COCODataset(
            cocoann_file=test_ann,
            image_processor=image_processor,
            image_root=test_img,
        )

        if batch_eval_metrics:
            trainer.compute_metrics = partial(
                compute_metrics_batch,
                image_processor=image_processor,
                cocoann_file=test_ann,
                size_map=create_size_map(cocoann_file=test_ann),
            )
        else:
            partial(
                compute_metrics,
                image_processor=image_processor,
                cocoann_file=test_ann,
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
    parser.add_argument(
        "--name",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Name of the experiment",
    )
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
