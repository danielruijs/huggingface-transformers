import torch
import argparse
import time
import json
import os
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from coco_utils import COCODataset
from torch.utils.data import DataLoader
from coco_utils import compute_COCO_metrics, create_size_map
from train import serialize_tensor_dict

EVAL_RESULTS_JSONL = "eval_results.jsonl"


def run_inference(
    image_processor, model, dataset, threshold, size_map, precision, use_lowmem
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
        if precision == "fp16":
            inputs = inputs.half()
        batch_labels = batch["labels"]

        # Forward pass
        if precision == "amp":
            start_forward = time.perf_counter()
            with torch.amp.autocast(model.device.type):
                with torch.no_grad():
                    outputs = model(inputs)
            end_forward = time.perf_counter()
        else:
            start_forward = time.perf_counter()
            with torch.no_grad():
                outputs = model(inputs)
            end_forward = time.perf_counter()

        # Post-processing of outputs
        start_post = time.perf_counter()
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=[size_map[int(batch_labels["image_id"])]],
            threshold=threshold,
        )
        end_post = time.perf_counter()

        if i >= warmup_batches:
            total_forward += end_forward - start_forward
            total_post += end_post - start_post

        if use_lowmem:
            with open(EVAL_RESULTS_JSONL, "a") as f:
                record = {
                    "predictions": serialize_tensor_dict(results[0]),
                    "labels": serialize_tensor_dict(batch_labels),
                }
                f.write(json.dumps(record) + "\n")
        else:
            predictions.extend(results)
            labels.append(batch_labels)

    avg_forward_time = total_forward / (len(dataloader) - warmup_batches)
    avg_post_time = total_post / (len(dataloader) - warmup_batches)
    print(
        f"\nTotal batches processed: {len(dataloader)} (of which {warmup_batches} were not included in timing)"
    )
    print(f"Average forward pass time: {avg_forward_time * 1000:.4f} ms")
    print(f"Average post-processing time: {avg_post_time * 1000:.4f} ms\n")

    if use_lowmem:
        with open(EVAL_RESULTS_JSONL, "r") as f:
            for line in f:
                rec = json.loads(line)
                predictions.append(rec["predictions"])
                labels.append(rec["labels"])
    return predictions, labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForObjectDetection.from_pretrained(args.model_dir).to(device)

    if args.precision == "fp16":
        model.half()

    dataset = COCODataset(
        cocoann_file=args.cocoann_file,
        image_processor=image_processor,
        image_root=args.img_dir,
    )

    predictions, labels = run_inference(
        image_processor=image_processor,
        model=model,
        dataset=dataset,
        threshold=args.threshold,
        size_map=create_size_map(cocoann_file=args.cocoann_file),
        precision=args.precision,
        use_lowmem=args.lowmem,
    )

    _ = compute_COCO_metrics(
        predictions=predictions, labels=labels, cocoann_file=args.cocoann_file
    )

    if os.path.exists(EVAL_RESULTS_JSONL):
        os.remove(EVAL_RESULTS_JSONL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory of the model checkpoint"
    )
    parser.add_argument(
        "--cocoann_file",
        type=str,
        required=True,
        help="Path to COCO annotations file",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="",
        help="Image directory. May be omitted if the annotation file contains full paths to the images.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "amp"],
        help="Precision options",
    )
    parser.add_argument("--lowmem", action="store_true", help="Use low memory mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
