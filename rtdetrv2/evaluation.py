import torch
import yaml
import argparse
import time
import os
from tqdm import tqdm
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_utils import COCODataset
from torch.utils.data import DataLoader


def run_inference(
    image_processor,
    model,
    dataset,
    threshold,
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    predictions = []
    labels = []
    total_forward = 0
    total_post = 0
    warmup_batches = 5  # Number of batches to not include in timing
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = batch["pixel_values"].to(model.device)
        batch_labels = batch["labels"]

        # Forward pass
        start_forward = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
        end_forward = time.perf_counter()

        # Post-processing of outputs
        start_post = time.perf_counter()
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=batch_labels["size"],
            threshold=threshold,
        )
        end_post = time.perf_counter()

        if i >= warmup_batches:
            total_forward += end_forward - start_forward
            total_post += end_post - start_post

        predictions.extend(results)
        labels.append(batch_labels)

    avg_forward_time = total_forward / (len(dataloader) - warmup_batches)
    avg_post_time = total_post / (len(dataloader) - warmup_batches)
    print(
        f"\nTotal batches processed: {len(dataloader)} (of which {warmup_batches} were not included in timing)"
    )
    print(f"Average forward pass time: {avg_forward_time * 1000:.4f} ms")
    print(f"Average post-processing time: {avg_post_time * 1000:.4f} ms\n")
    return predictions, labels


def compute_metrics(predictions, labels, cocoann_file):
    # Format the predictions to COCO format
    coco_predictions = []
    for pred, gt_label in zip(predictions, labels):
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            x1, y1, x2, y2 = box.tolist()
            coco_predictions.append(
                {
                    "image_id": int(gt_label["image_id"]),
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


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, config["model_dir"])
    cocoann_file = os.path.join(script_dir, config["cocoann_file"])
    img_dir = os.path.join(script_dir, config["img_dir"]) if "img_dir" in config else ""

    image_processor = RTDetrImageProcessor.from_pretrained(model_dir)
    model = RTDetrV2ForObjectDetection.from_pretrained(model_dir).to(device)

    dataset = COCODataset(
        cocoann_file=cocoann_file,
        image_processor=image_processor,
        image_root=img_dir,
    )

    predictions, labels = run_inference(
        image_processor=image_processor,
        model=model,
        dataset=dataset,
        threshold=config["threshold"],
    )

    _ = compute_metrics(
        predictions=predictions, labels=labels, cocoann_file=cocoann_file
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(config_path=args.config)
    main(config)
